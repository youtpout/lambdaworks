# Lambdaworks GPU

GPU backends for Lambdaworks.  Two backends are provided — Apple Metal and AMD
ROCm/HIP — with an identical Rust API so switching is a one-line feature flag
change.

---

## Data format (both backends)

The GPU API receives flat `u64` buffers:

- **scalars:** `[s0_l0, s0_l1, s0_l2, s0_l3, s1_l0, ...]`
- **points:**  `[x0_l0..x0_l3, y0_l0..y0_l3, z0_l0..z0_l3, x1_l0..]`

All limbs are little-endian. Point coordinates are Jacobian in the field's
Montgomery representation. Curves must satisfy `a = 0` (Pallas, Vesta).

---

## Metal backend (Apple Silicon / macOS)

### Prerequisites

- macOS with Xcode Command Line Tools installed.

### API

```rust
use lambdaworks_gpu::metal::pippenger_msm::{MetalPippengerMSM, PippengerMSMConfig};

let mut msm = MetalPippengerMSM::new_pallas()?;
let result  = msm.compute(&scalars, &points)?;
```

- `PippengerMSMConfig::pallas()` / `::vesta()` — pre-built configs
- `msm.config_for_num_points(n)` — optimal window/chunk sizes for `n` points

### Build

```bash
cargo build -p lambdaworks-gpu --features metal
```

### Tests

```bash
# Unit tests (no GPU required)
cargo test -p lambdaworks-gpu --features metal

# Integration correctness tests (Metal GPU required)
cargo test -p lambdaworks-gpu --features metal --test pippenger_msm_correctness
```

### Benchmarks

Compares Arkworks, CPU sequential Pippenger, CPU parallel Pippenger, and the
Metal GPU backend for Pallas and Vesta across sizes `2^12`, `2^18`, `2^22`.

```bash
cargo bench -p lambdaworks-gpu --features metal --bench pippenger_msm
```

---

## ROCm/HIP backend (AMD GPU)

### Prerequisites

- ROCm ≥ 5.0 installed (tested on ROCm 7.2).
- `libamdhip64.so` and `libhiprtc.so` on the library path (standard ROCm
  install at `/opt/rocm`).
- Set `ROCM_PATH` if ROCm is installed elsewhere:

```bash
export ROCM_PATH=/path/to/rocm
```

The HIP kernel is compiled at runtime by hipRTC — no `hipcc` build step needed.

### API

```rust
use lambdaworks_gpu::rocm::pippenger_msm::{HipPippengerMSM, HipPippengerMSMConfig};

let mut msm = HipPippengerMSM::new_pallas()?;
let result  = msm.compute(&scalars, &points)?;

// Or use prepare/compute_prepared to reuse GPU buffers across calls:
let prepared = msm.prepare(&scalars, &points)?;
let result   = msm.compute_prepared(&prepared)?;
```

- `HipPippengerMSMConfig::pallas()` / `::vesta()` — pre-built configs
- `msm.config_for_num_points(n)` — optimal window/chunk sizes for `n` points

### Build

```bash
cargo build -p lambdaworks-gpu --features rocm
```

### Tests

```bash
# Unit tests (no GPU required — Montgomery inverse, recoding, arithmetic)
cargo test -p lambdaworks-gpu --features rocm

# Integration correctness tests (AMD GPU + ROCm required)
cargo test -p lambdaworks-gpu --features rocm --test rocm_pippenger_msm_correctness

# Run a specific test
cargo test -p lambdaworks-gpu --features rocm --test rocm_pippenger_msm_correctness \
    pallas_lambdaworks_cpu_and_hip_pippenger_match
```

### Benchmarks

Same benchmark file as Metal — both backends emit results in the same Criterion
groups for direct comparison.

```bash
cargo bench -p lambdaworks-gpu --features rocm --bench pippenger_msm
```

To run both backends simultaneously (if you have both an Apple GPU and an AMD
GPU on the same machine):

```bash
cargo bench -p lambdaworks-gpu --features metal,rocm --bench pippenger_msm
```

The benchmark sizes are `2^12`, `2^18`, `2^22` (defined in
`benches/pippenger_msm.rs`).

### Measured performance (RX 6800, gfx1030, ROCm 7.2)

| Variante | 2^18 (262 144 pts) |
|---|---|
| Arkworks `VariableBaseMSM` (CPU) | ~106 ms |
| CPU signed Pippenger (sequential) | ~1 600 ms |
| CPU signed Pippenger (parallel, 16-core) | ~164 ms |
| **HIP Pippenger (RX 6800)** | **~88 ms** |
