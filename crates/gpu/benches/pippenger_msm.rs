//! Criterion benchmarks for the Pippenger MSM GPU backends.
//!
//! Each benchmark group contains several variants:
//!   • `arkworks-variable-base`        – Arkworks reference
//!   • `cpu-signed-pippenger`          – lambdaworks single-threaded CPU
//!   • `cpu-parallel-signed-pippenger` – lambdaworks parallel CPU
//!   • `metal-pippenger`               – Apple Metal GPU  (feature = "metal")
//!   • `hip-pippenger`                 – AMD ROCm/HIP GPU (feature = "rocm")
//!
//! Run with:
//!   cargo bench --features metal   # Apple GPU
//!   cargo bench --features rocm    # AMD GPU
//!   cargo bench --features metal,rocm  # both

use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{BigInteger256, PrimeField, UniformRand};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::{
            curves::{pallas::curve::PallasCurve, vesta::curve::VestaCurve},
            point::ShortWeierstrassProjectivePoint,
            traits::IsShortWeierstrass,
        },
        traits::IsEllipticCurve,
    },
    field::{
        element::FieldElement,
        traits::{IsField, IsPrimeField},
    },
    msm::pippenger,
    unsigned_integer::element::UnsignedInteger,
};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

#[cfg(feature = "metal")]
use lambdaworks_gpu::metal::pippenger_msm::{MetalPippengerMSM, PippengerMSMConfig};

#[cfg(feature = "rocm")]
use lambdaworks_gpu::rocm::pippenger_msm::HipPippengerMSM;

const SEED: [u8; 32] = [0x42; 32];
const BENCH_SIZES: &[usize] = &[1 << 12, 1 << 18, 1 << 22];

type Scalar = UnsignedInteger<4>;
type LwPallasPoint = ShortWeierstrassProjectivePoint<PallasCurve>;
type LwVestaPoint = ShortWeierstrassProjectivePoint<VestaCurve>;

// ─── benchmark data ───────────────────────────────────────────────────────────

struct BenchData<ArkAffine, LwC>
where
    ArkAffine: AffineRepr,
    ArkAffine::ScalarField: PrimeField<BigInt = BigInteger256>,
    LwC: IsShortWeierstrass,
{
    ark_points: Vec<ArkAffine>,
    ark_scalars: Vec<ArkAffine::ScalarField>,
    lw_scalars: Vec<Scalar>,
    lw_points: Vec<ShortWeierstrassProjectivePoint<LwC>>,
    gpu_scalars: Vec<u64>,
    gpu_points: Vec<u64>,
}

fn pallas_data(size: usize) -> BenchData<mina_curves::pasta::Pallas, PallasCurve> {
    make_data::<mina_curves::pasta::ProjectivePallas, PallasCurve>(size, mina_pallas_generator())
}

fn vesta_data(size: usize) -> BenchData<mina_curves::pasta::Vesta, VestaCurve> {
    make_data::<mina_curves::pasta::ProjectiveVesta, VestaCurve>(size, mina_vesta_generator())
}

fn make_data<ArkG, LwC>(
    size: usize,
    lw_generator: ShortWeierstrassProjectivePoint<LwC>,
) -> BenchData<ArkG::Affine, LwC>
where
    ArkG: CurveGroup,
    ArkG::Affine: AffineRepr<Group = ArkG>,
    ArkG::ScalarField: PrimeField<BigInt = BigInteger256> + UniformRand,
    LwC: IsShortWeierstrass
        + IsEllipticCurve<PointRepresentation = ShortWeierstrassProjectivePoint<LwC>>,
    LwC::BaseField: IsField<BaseType = Scalar> + IsPrimeField<CanonicalType = Scalar>,
{
    let mut rng = ChaCha20Rng::from_seed(SEED);
    let ark_generator = ArkG::generator();
    let mut ark_points = Vec::with_capacity(size);
    let mut ark_scalars = Vec::with_capacity(size);
    let mut lw_points = Vec::with_capacity(size);
    let mut lw_scalars = Vec::with_capacity(size);

    for _ in 0..size {
        let scalar = ArkG::ScalarField::rand(&mut rng);
        let point_scalar = ArkG::ScalarField::rand(&mut rng);
        let scalar_bigint = scalar.into_bigint();
        let point_scalar_bigint = point_scalar.into_bigint();

        ark_scalars.push(scalar);
        ark_points.push((ark_generator * point_scalar).into_affine());
        lw_scalars.push(unsigned_from_bigint(scalar_bigint));
        lw_points.push(lw_generator.operate_with_self(unsigned_from_bigint(point_scalar_bigint)));
    }

    let gpu_scalars = encode_scalars(&lw_scalars);
    let gpu_points = encode_points(&lw_points);

    BenchData {
        ark_points,
        ark_scalars,
        lw_scalars,
        lw_points,
        gpu_scalars,
        gpu_points,
    }
}

// ─── Pallas benchmarks ────────────────────────────────────────────────────────

fn bench_pallas(c: &mut Criterion) {
    let mut group = c.benchmark_group("pallas-pippenger-msm");

    for &size in BENCH_SIZES {
        let data = pallas_data(size);
        let window_size = optimal_window_size(size);

        group.bench_with_input(
            BenchmarkId::new("arkworks-variable-base", size),
            &data,
            |b, data| b.iter(|| black_box(ark_msm(&data.ark_points, &data.ark_scalars))),
        );

        group.bench_with_input(
            BenchmarkId::new("cpu-parallel-signed-pippenger", size),
            &data,
            |b, data| {
                b.iter(|| {
                    black_box(pippenger::parallel_msm_with_signed(
                        &data.lw_scalars,
                        &data.lw_points,
                        window_size,
                    ))
                })
            },
        );

        #[cfg(feature = "metal")]
        {
            let base_msm = MetalPippengerMSM::new_pallas().expect("Metal device required");
            let config = base_msm.config_for_num_points(size);
            let mut msm = MetalPippengerMSM::new(config).expect("Metal device required");
            let prepared = msm
                .prepare(&data.gpu_scalars, &data.gpu_points)
                .expect("Metal prepare failed");

            group.bench_with_input(
                BenchmarkId::new("metal-pippenger", size),
                &data,
                |b, _| {
                    b.iter(|| {
                        black_box(msm.compute_prepared(&prepared).expect("Metal MSM failed"))
                    })
                },
            );
        }

        #[cfg(feature = "rocm")]
        {
            let base_msm = HipPippengerMSM::new_pallas().expect("ROCm device required");
            let config = base_msm.config_for_num_points(size);
            let mut msm = HipPippengerMSM::new(config).expect("ROCm device required");
            let prepared = msm
                .prepare(&data.gpu_scalars, &data.gpu_points)
                .expect("HIP prepare failed");

            group.bench_with_input(
                BenchmarkId::new("hip-pippenger", size),
                &data,
                |b, _| {
                    b.iter(|| {
                        black_box(msm.compute_prepared(&prepared).expect("HIP MSM failed"))
                    })
                },
            );
        }
    }

    group.finish();
}

// ─── Vesta benchmarks ─────────────────────────────────────────────────────────

fn bench_vesta(c: &mut Criterion) {
    let mut group = c.benchmark_group("vesta-pippenger-msm");

    for &size in BENCH_SIZES {
        let data = vesta_data(size);
        let window_size = optimal_window_size(size);

        group.bench_with_input(
            BenchmarkId::new("arkworks-variable-base", size),
            &data,
            |b, data| b.iter(|| black_box(ark_msm(&data.ark_points, &data.ark_scalars))),
        );

        group.bench_with_input(
            BenchmarkId::new("cpu-parallel-signed-pippenger", size),
            &data,
            |b, data| {
                b.iter(|| {
                    black_box(pippenger::parallel_msm_with_signed(
                        &data.lw_scalars,
                        &data.lw_points,
                        window_size,
                    ))
                })
            },
        );

        #[cfg(feature = "metal")]
        {
            let base_msm = MetalPippengerMSM::new_vesta().expect("Metal device required");
            let config = base_msm.config_for_num_points(size);
            let mut msm = MetalPippengerMSM::new(config).expect("Metal device required");
            let prepared = msm
                .prepare(&data.gpu_scalars, &data.gpu_points)
                .expect("Metal prepare failed");

            group.bench_with_input(
                BenchmarkId::new("metal-pippenger", size),
                &data,
                |b, _| {
                    b.iter(|| {
                        black_box(msm.compute_prepared(&prepared).expect("Metal MSM failed"))
                    })
                },
            );
        }

        #[cfg(feature = "rocm")]
        {
            let base_msm = HipPippengerMSM::new_vesta().expect("ROCm device required");
            let config = base_msm.config_for_num_points(size);
            let mut msm = HipPippengerMSM::new(config).expect("ROCm device required");
            let prepared = msm
                .prepare(&data.gpu_scalars, &data.gpu_points)
                .expect("HIP prepare failed");

            group.bench_with_input(
                BenchmarkId::new("hip-pippenger", size),
                &data,
                |b, _| {
                    b.iter(|| {
                        black_box(msm.compute_prepared(&prepared).expect("HIP MSM failed"))
                    })
                },
            );
        }
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10).warm_up_time(Duration::from_secs(1)).measurement_time(Duration::from_secs(10));
    targets = bench_pallas, bench_vesta
}
criterion_main!(benches);

// ─── shared helpers ───────────────────────────────────────────────────────────

fn optimal_window_size(n: usize) -> usize {
    match n {
        0..=4 => 2,
        5..=32 => 4,
        33..=256 => 6,
        257..=4096 => 7,
        _ => 9,
    }
}

fn ark_msm<G>(points: &[G], scalars: &[G::ScalarField]) -> G::Group
where
    G: AffineRepr,
    G::Group: VariableBaseMSM<MulBase = G>,
    G::ScalarField: PrimeField<BigInt = BigInteger256>,
{
    <G::Group as VariableBaseMSM>::msm(points, scalars).expect("valid Arkworks MSM input")
}

fn encode_scalars(scalars: &[Scalar]) -> Vec<u64> {
    let mut out = Vec::with_capacity(scalars.len() * 4);
    for s in scalars {
        out.extend_from_slice(&little_endian_limbs(s));
    }
    out
}

fn encode_points<C>(points: &[ShortWeierstrassProjectivePoint<C>]) -> Vec<u64>
where
    C: IsShortWeierstrass,
    C::BaseField: IsField<BaseType = Scalar>,
{
    let mut out = Vec::with_capacity(points.len() * 12);
    for p in points {
        let affine = p.to_affine();
        out.extend_from_slice(&little_endian_limbs(affine.x().value()));
        out.extend_from_slice(&little_endian_limbs(affine.y().value()));
        out.extend_from_slice(&little_endian_limbs(affine.z().value()));
    }
    out
}

fn mina_pallas_generator() -> LwPallasPoint {
    let x = FieldElement::from(&Scalar::from_u64(1));
    let y = FieldElement::from_hex(
        "1b74b5a30a12937c53dfa9f06378ee548f655bd4333d477119cf7a23caed2abb",
    )
    .expect("valid y");
    PallasCurve::create_point_from_affine(x, y).expect("valid Pallas generator")
}

fn mina_vesta_generator() -> LwVestaPoint {
    let x = FieldElement::from(&Scalar::from_u64(1));
    let y = FieldElement::from_hex(
        "1943666ea922ae6b13b64e3aae89754cacce3a7f298ba20c4e4389b9b0276a62",
    )
    .expect("valid y");
    VestaCurve::create_point_from_affine(x, y).expect("valid Vesta generator")
}

fn unsigned_from_bigint(v: BigInteger256) -> Scalar {
    Scalar::from_limbs([v.0[3], v.0[2], v.0[1], v.0[0]])
}

fn little_endian_limbs(v: &Scalar) -> [u64; 4] {
    [v.limbs[3], v.limbs[2], v.limbs[1], v.limbs[0]]
}
