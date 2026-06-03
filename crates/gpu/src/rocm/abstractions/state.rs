//! ROCm/HIP state management with hipRTC runtime compilation.
//!
//! Uses the hipRTC API to compile HIP kernels from source at runtime,
//! mirroring the approach of `DynamicMetalState` on Apple platforms.
//! The compiled bitcode is loaded via the hipModule API.

use std::collections::HashMap;
use std::ffi::{CStr, CString};

use super::errors::{HipError, HipResult};

// ─── hipRTC / HIP module opaque handle types ─────────────────────────────────

/// `hiprtcProgram` is opaque; we store it as a raw pointer.
type HiprtcProgram = *mut std::ffi::c_void;

/// `hipModule_t` – handle returned by `hipModuleLoadData`.
type HipModule = *mut std::ffi::c_void;

/// `hipFunction_t` – handle returned by `hipModuleGetFunction`.
type HipFunction = *mut std::ffi::c_void;

/// `hipStream_t` – NULL means the default stream.
type HipStream = *mut std::ffi::c_void;

const HIP_MEMCPY_HOST_TO_DEVICE: u32 = 1;
const HIP_MEMCPY_DEVICE_TO_HOST: u32 = 2;

// ─── extern declarations ──────────────────────────────────────────────────────
//
// These link against `amdhip64` (HIP runtime) and `hiprtc` (runtime compiler).
// On a standard ROCm installation both live in `/opt/rocm/lib/`.

extern "C" {
    fn hipGetDeviceCount(count: *mut i32) -> u32;
    fn hipSetDevice(device: i32) -> u32;
    fn hipGetErrorString(error: u32) -> *const std::os::raw::c_char;
    fn hipMalloc(ptr: *mut *mut std::ffi::c_void, size: usize) -> u32;
    fn hipFree(ptr: *mut std::ffi::c_void) -> u32;
    fn hipMemcpy(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        size: usize,
        kind: u32,
    ) -> u32;
    fn hipMemset(ptr: *mut std::ffi::c_void, value: i32, size: usize) -> u32;
    fn hipDeviceSynchronize() -> u32;

    // hipRTC
    fn hiprtcCreateProgram(
        prog: *mut HiprtcProgram,
        src: *const std::os::raw::c_char,
        name: *const std::os::raw::c_char,
        num_headers: i32,
        headers: *const *const std::os::raw::c_char,
        include_names: *const *const std::os::raw::c_char,
    ) -> u32;
    fn hiprtcCompileProgram(
        prog: HiprtcProgram,
        num_options: i32,
        options: *const *const std::os::raw::c_char,
    ) -> u32;
    fn hiprtcGetProgramLogSize(prog: HiprtcProgram, log_size: *mut usize) -> u32;
    fn hiprtcGetProgramLog(prog: HiprtcProgram, log: *mut std::os::raw::c_char) -> u32;
    // hiprtcGetCode returns the compiled binary (HSACO on AMD) — suitable for hipModuleLoadData.
    fn hiprtcGetCodeSize(prog: HiprtcProgram, size: *mut usize) -> u32;
    fn hiprtcGetCode(prog: HiprtcProgram, code: *mut std::os::raw::c_char) -> u32;
    fn hiprtcDestroyProgram(prog: *mut HiprtcProgram) -> u32;

    // HIP module API
    fn hipModuleLoadData(module: *mut HipModule, image: *const std::ffi::c_void) -> u32;
    fn hipModuleUnload(module: HipModule) -> u32;
    fn hipModuleGetFunction(
        function: *mut HipFunction,
        module: HipModule,
        name: *const std::os::raw::c_char,
    ) -> u32;
    fn hipModuleLaunchKernel(
        f: HipFunction,
        grid_dim_x: u32,
        grid_dim_y: u32,
        grid_dim_z: u32,
        block_dim_x: u32,
        block_dim_y: u32,
        block_dim_z: u32,
        shared_mem_bytes: u32,
        stream: HipStream,
        kernel_params: *mut *mut std::ffi::c_void,
        extra: *mut *mut std::ffi::c_void,
    ) -> u32;
}

// ─── helper ───────────────────────────────────────────────────────────────────

fn hip_err(code: u32) -> HipError {
    let msg = unsafe {
        let ptr = hipGetErrorString(code);
        if ptr.is_null() {
            format!("unknown HIP error {}", code)
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };
    HipError::Runtime { code, msg }
}

macro_rules! hip_check {
    ($expr:expr) => {{
        let code = unsafe { $expr };
        if code != 0 {
            return Err(hip_err(code));
        }
    }};
}

// ─── DeviceBuffer ─────────────────────────────────────────────────────────────

/// RAII wrapper around a HIP device allocation.
pub struct DeviceBuffer {
    ptr: *mut std::ffi::c_void,
    len: usize, // bytes
}

impl DeviceBuffer {
    pub fn len_bytes(&self) -> usize {
        self.len
    }

    pub(crate) fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { hipFree(self.ptr) };
        }
    }
}

// ─── HipState ─────────────────────────────────────────────────────────────────

/// ROCm GPU state manager with hipRTC runtime compilation.
///
/// Mirrors `DynamicMetalState`: compile a kernel source string once, then
/// launch named kernels via `execute_compute` / `execute_compute_seq`.
pub struct HipState {
    module: Option<HipModule>,
    functions: HashMap<String, HipFunction>,
}

impl HipState {
    /// Initialise the HIP runtime and select the default device (device 0).
    pub fn new() -> HipResult<Self> {
        let mut count = 0i32;
        hip_check!(hipGetDeviceCount(&mut count));
        if count == 0 {
            return Err(HipError::DeviceNotFound);
        }
        hip_check!(hipSetDevice(0));
        Ok(Self {
            module: None,
            functions: HashMap::new(),
        })
    }

    /// Compile `source` with hipRTC and load the resulting bitcode module.
    ///
    /// Must be called before any kernel is launched. `name` is used as the
    /// program identifier in hipRTC error messages.
    pub fn load_source(&mut self, source: &str, name: &str) -> HipResult<()> {
        let src_c = CString::new(source).expect("source contains null byte");
        let name_c = CString::new(name).expect("name contains null byte");

        // Create the hipRTC program.
        let mut prog: HiprtcProgram = std::ptr::null_mut();
        {
            let code = unsafe {
                hiprtcCreateProgram(
                    &mut prog,
                    src_c.as_ptr(),
                    name_c.as_ptr(),
                    0,
                    std::ptr::null(),
                    std::ptr::null(),
                )
            };
            if code != 0 {
                return Err(HipError::CompilationError(format!(
                    "hiprtcCreateProgram failed with code {}",
                    code
                )));
            }
        }

        // Compile.
        let compile_code =
            unsafe { hiprtcCompileProgram(prog, 0, std::ptr::null()) };

        if compile_code != 0 {
            let log = self.get_rtc_log(prog);
            unsafe { hiprtcDestroyProgram(&mut prog) };
            return Err(HipError::CompilationError(log));
        }

        // Extract the compiled binary (HSACO on AMD — directly loadable by hipModuleLoadData).
        let mut code_size: usize = 0;
        unsafe { hiprtcGetCodeSize(prog, &mut code_size) };
        let mut code = vec![0u8; code_size];
        unsafe {
            hiprtcGetCode(prog, code.as_mut_ptr() as *mut std::os::raw::c_char)
        };
        unsafe { hiprtcDestroyProgram(&mut prog) };

        // Load the HSACO binary as a HIP module.
        let mut module: HipModule = std::ptr::null_mut();
        {
            let code = unsafe {
                hipModuleLoadData(&mut module, code.as_ptr() as *const std::ffi::c_void)
            };
            if code != 0 {
                return Err(HipError::ModuleLoadError(format!(
                    "hipModuleLoadData failed with code {}",
                    code
                )));
            }
        }

        if let Some(old) = self.module.take() {
            unsafe { hipModuleUnload(old) };
        }
        self.module = Some(module);
        self.functions.clear();
        Ok(())
    }

    /// Pre-load a kernel function by name; returns the block size hint (256).
    pub fn prepare_function(&mut self, name: &str) -> HipResult<u32> {
        let _ = self.get_function(name)?;
        Ok(256)
    }

    fn get_function(&mut self, name: &str) -> HipResult<HipFunction> {
        if let Some(&f) = self.functions.get(name) {
            return Ok(f);
        }
        let module = self
            .module
            .ok_or_else(|| HipError::FunctionError("no module loaded".to_string()))?;
        let name_c = CString::new(name).expect("name contains null byte");
        let mut func: HipFunction = std::ptr::null_mut();
        let code =
            unsafe { hipModuleGetFunction(&mut func, module, name_c.as_ptr()) };
        if code != 0 {
            return Err(HipError::FunctionError(name.to_string()));
        }
        self.functions.insert(name.to_string(), func);
        Ok(func)
    }

    // ─── buffer helpers ────────────────────────────────────────────────────

    /// Allocate `bytes` bytes of uninitialised device memory.
    pub fn alloc_buffer(&self, bytes: usize) -> HipResult<DeviceBuffer> {
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let code = unsafe { hipMalloc(&mut ptr, bytes) };
        if code != 0 {
            return Err(HipError::AllocationError(format!("code {}", code)));
        }
        Ok(DeviceBuffer { ptr, len: bytes })
    }

    /// Allocate a device buffer and upload `data` from the host.
    pub fn alloc_buffer_with_data<T: Copy>(&self, data: &[T]) -> HipResult<DeviceBuffer> {
        let bytes = std::mem::size_of_val(data);
        let buf = self.alloc_buffer(bytes)?;
        let code = unsafe {
            hipMemcpy(
                buf.ptr,
                data.as_ptr() as *const std::ffi::c_void,
                bytes,
                HIP_MEMCPY_HOST_TO_DEVICE,
            )
        };
        if code != 0 {
            return Err(HipError::MemcpyError(format!("H->D failed, code {}", code)));
        }
        Ok(buf)
    }

    /// Download `count` elements of type `T` from a device buffer.
    pub fn read_buffer<T: Copy + Default>(&self, buf: &DeviceBuffer, count: usize) -> HipResult<Vec<T>> {
        let bytes = count * std::mem::size_of::<T>();
        let mut host = vec![T::default(); count];
        let code = unsafe {
            hipMemcpy(
                host.as_mut_ptr() as *mut std::ffi::c_void,
                buf.ptr,
                bytes,
                HIP_MEMCPY_DEVICE_TO_HOST,
            )
        };
        if code != 0 {
            return Err(HipError::MemcpyError(format!("D->H failed, code {}", code)));
        }
        Ok(host)
    }

    /// Zero a device buffer.
    pub fn zero_buffer(&self, buf: &DeviceBuffer) -> HipResult<()> {
        let code = unsafe { hipMemset(buf.ptr, 0, buf.len) };
        if code != 0 {
            return Err(HipError::MemcpyError(format!("hipMemset failed, code {}", code)));
        }
        Ok(())
    }

    // ─── kernel launch ─────────────────────────────────────────────────────

    /// Launch `kernel_name` with the given buffers and 1-D thread grid.
    ///
    /// `thread_count` is the total number of threads; they are grouped in
    /// blocks of 256 (matching the Metal implementation).
    pub fn execute_compute(
        &mut self,
        kernel_name: &str,
        buffers: &[&DeviceBuffer],
        thread_count: u64,
    ) -> HipResult<()> {
        self.execute_compute_seq(&[(kernel_name, buffers, thread_count)])
    }

    /// Launch multiple kernels sequentially using the same HIP stream (null = default).
    ///
    /// All kernels are launched before `hipDeviceSynchronize`, minimising
    /// host-device round-trips.
    pub fn execute_compute_seq(
        &mut self,
        kernels: &[(&str, &[&DeviceBuffer], u64)],
    ) -> HipResult<()> {
        self.execute_compute_seq_2d(kernels, 1)
    }

    /// Like `execute_compute_seq` but with a 2-D grid: `grid_dim_x × grid_y`.
    ///
    /// `grid_y` is applied uniformly to every kernel in the sequence.
    /// Use `grid_y = 1` for the single-MSM (non-batched) path.
    pub fn execute_compute_seq_2d(
        &mut self,
        kernels: &[(&str, &[&DeviceBuffer], u64)],
        grid_y: u32,
    ) -> HipResult<()> {
        for (name, buffers, thread_count) in kernels {
            let func = self.get_function(name)?;
            let block_size: u32 = 256;
            let grid_x: u32 = ((*thread_count as u32) + block_size - 1) / block_size;

            // Build the kernel-params array: one void* per buffer (device ptr).
            let mut ptrs: Vec<*mut std::ffi::c_void> =
                buffers.iter().map(|b| b.ptr).collect();
            let mut kernel_params: Vec<*mut std::ffi::c_void> =
                ptrs.iter_mut()
                    .map(|p| p as *mut *mut std::ffi::c_void as *mut std::ffi::c_void)
                    .collect();

            let code = unsafe {
                hipModuleLaunchKernel(
                    func,
                    grid_x, grid_y, 1,
                    block_size, 1, 1,
                    0,
                    std::ptr::null_mut(), // default stream
                    kernel_params.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            };
            if code != 0 {
                return Err(HipError::LaunchError(format!(
                    "kernel '{}' failed with code {}",
                    name, code
                )));
            }
        }

        // Synchronise once after all kernel launches.
        hip_check!(hipDeviceSynchronize());
        Ok(())
    }

    // ─── internal ──────────────────────────────────────────────────────────

    fn get_rtc_log(&self, prog: HiprtcProgram) -> String {
        let mut log_size: usize = 0;
        unsafe { hiprtcGetProgramLogSize(prog, &mut log_size) };
        if log_size == 0 {
            return "(no log)".to_string();
        }
        let mut log = vec![0u8; log_size];
        unsafe { hiprtcGetProgramLog(prog, log.as_mut_ptr() as *mut std::os::raw::c_char) };
        String::from_utf8_lossy(&log).trim_end_matches('\0').to_string()
    }
}

impl Drop for HipState {
    fn drop(&mut self) {
        if let Some(module) = self.module.take() {
            unsafe { hipModuleUnload(module) };
        }
    }
}
