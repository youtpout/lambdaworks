# Lambdaworks GPU

GPU backends for Lambdaworks.

## Metal Pippenger MSM

The Metal backend includes a dedicated Pippenger MSM implementation for 256-bit
short-Weierstrass curves with `a = 0`, currently configured for Pallas and Vesta:

- `lambdaworks_gpu::metal::pippenger_msm::MetalPippengerMSM`
- `lambdaworks_gpu::metal::pippenger_msm::PippengerMSMConfig`
- `PippengerMSMConfig::pallas()`
- `PippengerMSMConfig::vesta()`

The existing `metal::msm` module is left untouched. The dedicated Pippenger MSM
lives in `metal::pippenger_msm`.

### Data Format

The GPU API receives flat `u64` buffers:

- scalars: `[s0_l0, s0_l1, s0_l2, s0_l3, s1_l0, ...]`
- points: `[x0_l0..x0_l3, y0_l0..y0_l3, z0_l0..z0_l3, x1_l0..]`

All limbs are little-endian. Point coordinates are Jacobian coordinates in the
field's Montgomery representation.

### Correctness Tests

The integration tests generate deterministic Pallas and Vesta inputs. They use
the Mina Pasta generator for both Arkworks and Lambdaworks, then check:

- Arkworks `VariableBaseMSM` equals Lambdaworks CPU signed Pippenger.
- Lambdaworks CPU signed Pippenger equals Metal Pippenger.

```bash
cargo test -p lambdaworks-gpu --features metal --test pippenger_msm_correctness
```

### Benchmarks

The Criterion benchmark compares:

- Arkworks `VariableBaseMSM`
- CPU `lambdaworks_math::msm::pippenger::msm_with_signed`
- CPU `parallel_msm_with_signed`
- GPU `MetalPippengerMSM`

The benchmark performs Arkworks/Lambdaworks and CPU/GPU equality checks before
measuring each curve and size.

```bash
cargo bench -p lambdaworks-gpu --features metal --bench pippenger_msm
```

The benchmark sizes are defined in `crates/gpu/benches/pippenger_msm.rs`.
