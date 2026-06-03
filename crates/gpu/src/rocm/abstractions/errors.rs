use thiserror::Error;

/// Errors that can occur during ROCm/HIP GPU operations.
#[derive(Debug, Error)]
pub enum HipError {
    /// No HIP-compatible AMD GPU device was found.
    #[error("HIP device not found - ensure you're running on a ROCm-compatible AMD GPU")]
    DeviceNotFound,

    /// HIP runtime returned a non-zero status code.
    #[error("HIP runtime error {code}: {msg}")]
    Runtime { code: u32, msg: String },

    /// hipRTC compilation failed.
    #[error("hipRTC compilation failed: {0}")]
    CompilationError(String),

    /// Failed to load a HIP module from compiled bitcode.
    #[error("Failed to load HIP module: {0}")]
    ModuleLoadError(String),

    /// Kernel function not found in the compiled module.
    #[error("Failed to get HIP kernel function '{0}'")]
    FunctionError(String),

    /// Failed to allocate device memory.
    #[error("Failed to allocate HIP device memory: {0}")]
    AllocationError(String),

    /// Failed to copy memory between host and device.
    #[error("HIP memcpy failed: {0}")]
    MemcpyError(String),

    /// Kernel launch failed.
    #[error("HIP kernel launch failed: {0}")]
    LaunchError(String),

    /// Input size is invalid for the operation.
    #[error("Invalid input size: expected multiple of {expected}, got {actual}")]
    InvalidInputSize { expected: usize, actual: usize },

    /// MSM scalar/point count mismatch.
    #[error("MSM length mismatch: {0} scalars vs {1} points")]
    LengthMismatch(usize, usize),

    /// MSM received empty input.
    #[error("MSM received empty input")]
    EmptyInput,
}

pub type HipResult<T> = Result<T, HipError>;
