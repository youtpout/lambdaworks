//! ROCm/HIP GPU backend for lambdaworks.
//!
//! Provides GPU acceleration for AMD GPUs via the ROCm/HIP runtime and the
//! hipRTC just-in-time compiler.  The API mirrors the Metal backend so that
//! users can switch between Apple and AMD GPUs with minimal code changes.
//!
//! # Supported operations
//!
//! - Multi-Scalar Multiplication (MSM) using Pippenger's algorithm
//!   [`pippenger_msm::HipPippengerMSM`]
//!
//! # Prerequisites
//!
//! - A ROCm-compatible AMD GPU (RDNA or CDNA architecture).
//! - ROCm ≥ 5.0 installed at `/opt/rocm` (or `ROCM_PATH` set accordingly).
//! - `libamdhip64.so` and `libhiprtc.so` available on the linker path.
//!
//! # Usage
//!
//! ```toml
//! [dependencies]
//! lambdaworks-gpu = { version = "...", features = ["rocm"] }
//! ```

pub mod abstractions;
pub mod pippenger_msm;
