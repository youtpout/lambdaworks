#![cfg(feature = "metal")]

use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{BigInteger256, PrimeField, UniformRand};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use lambdaworks_gpu::metal::pippenger_msm::{MetalPippengerMSM, PippengerMSMConfig};
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

const SEED: [u8; 32] = [0x42; 32];
const BENCH_SIZES: &[usize] = &[1 << 12, 1 << 18, 1 << 22];

type Scalar = UnsignedInteger<4>;
type LwPallasPoint = ShortWeierstrassProjectivePoint<PallasCurve>;
type LwVestaPoint = ShortWeierstrassProjectivePoint<VestaCurve>;

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

    let gpu_scalars = gpu_scalars(&lw_scalars);
    let gpu_points = gpu_points(&lw_points);

    BenchData {
        ark_points,
        ark_scalars,
        lw_scalars,
        lw_points,
        gpu_scalars,
        gpu_points,
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

fn bench_curve<ArkAffine, LwC>(
    c: &mut Criterion,
    group_name: &str,
    make_data_for_curve: impl Fn(usize) -> BenchData<ArkAffine, LwC>,
    config_for_curve: impl Fn() -> PippengerMSMConfig,
) where
    ArkAffine: AffineRepr + 'static,
    ArkAffine::BaseField: PrimeField<BigInt = BigInteger256>,
    ArkAffine::Group: VariableBaseMSM<MulBase = ArkAffine>,
    ArkAffine::ScalarField: PrimeField<BigInt = BigInteger256>,
    LwC: IsShortWeierstrass
        + IsEllipticCurve<PointRepresentation = ShortWeierstrassProjectivePoint<LwC>>
        + 'static,
    LwC::BaseField: IsField<BaseType = Scalar> + IsPrimeField<CanonicalType = Scalar>,
{
    let mut group = c.benchmark_group(group_name);

    for &size in BENCH_SIZES {
        let data = make_data_for_curve(size);
        let base_msm = MetalPippengerMSM::new(config_for_curve()).expect("Metal device required");
        let config = base_msm.config_for_num_points(size);

        assert_arkworks_lambdaworks_match(&data, config.window_size);
        let cpu_expected =
            pippenger::msm_with_signed(&data.lw_scalars, &data.lw_points, config.window_size)
                .to_affine();
        let mut msm = MetalPippengerMSM::new(config).expect("Metal device required");
        let prepared = msm
            .prepare(&data.gpu_scalars, &data.gpu_points)
            .expect("Metal Pippenger MSM preparation failed");
        let gpu_result = msm
            .compute_prepared(&prepared)
            .expect("Metal Pippenger MSM failed during correctness check");
        let gpu_result = gpu_result_to_point::<LwC>(&gpu_result).to_affine();
        assert_eq!(
            gpu_result, cpu_expected,
            "{group_name} GPU mismatch for size={size}"
        );

        group.bench_with_input(
            BenchmarkId::new("arkworks-variable-base", size),
            &data,
            |b, data| b.iter(|| black_box(ark_msm(&data.ark_points, &data.ark_scalars))),
        );

        group.bench_with_input(
            BenchmarkId::new("cpu-signed-pippenger", size),
            &data,
            |b, _data| {
                b.iter(|| {
                    black_box(pippenger::msm_with_signed(
                        &data.lw_scalars,
                        &data.lw_points,
                        PippengerMSMConfig::optimal_window_size(data.lw_scalars.len()),
                    ))
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("cpu-parallel-signed-pippenger", size),
            &data,
            |b, _data| {
                b.iter(|| {
                    black_box(pippenger::parallel_msm_with_signed(
                        &data.lw_scalars,
                        &data.lw_points,
                        PippengerMSMConfig::optimal_window_size(data.lw_scalars.len()),
                    ))
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("metal-pippenger", size),
            &data,
            |b, _data| {
                b.iter(|| {
                    black_box(
                        msm.compute_prepared(&prepared)
                            .expect("Metal Pippenger MSM failed"),
                    )
                })
            },
        );
    }

    group.finish();
}

fn assert_arkworks_lambdaworks_match<ArkAffine, LwC>(
    data: &BenchData<ArkAffine, LwC>,
    window_size: usize,
) where
    ArkAffine: AffineRepr,
    ArkAffine::BaseField: PrimeField<BigInt = BigInteger256>,
    ArkAffine::Group: VariableBaseMSM<MulBase = ArkAffine>,
    ArkAffine::ScalarField: PrimeField<BigInt = BigInteger256>,
    LwC: IsShortWeierstrass
        + IsEllipticCurve<PointRepresentation = ShortWeierstrassProjectivePoint<LwC>>,
    LwC::BaseField: IsField<BaseType = Scalar> + IsPrimeField<CanonicalType = Scalar>,
{
    let ark = ark_msm(&data.ark_points, &data.ark_scalars).into_affine();
    let lw = pippenger::msm_with_signed(&data.lw_scalars, &data.lw_points, window_size).to_affine();

    assert!(
        field_eq(ark.x().expect("non-infinity Arkworks MSM result"), lw.x()),
        "MSM x-coordinate mismatch"
    );
    assert!(
        field_eq(ark.y().expect("non-infinity Arkworks MSM result"), lw.y()),
        "MSM y-coordinate mismatch"
    );
}

fn gpu_scalars(scalars: &[Scalar]) -> Vec<u64> {
    let mut out = Vec::with_capacity(scalars.len() * 4);
    for scalar in scalars {
        out.extend_from_slice(&little_endian_limbs(scalar));
    }
    out
}

fn gpu_points<C>(points: &[ShortWeierstrassProjectivePoint<C>]) -> Vec<u64>
where
    C: IsShortWeierstrass,
    C::BaseField: IsField<BaseType = Scalar>,
{
    let mut out = Vec::with_capacity(points.len() * 12);
    for point in points {
        let affine = point.to_affine();
        out.extend_from_slice(&little_endian_limbs(affine.x().value()));
        out.extend_from_slice(&little_endian_limbs(affine.y().value()));
        out.extend_from_slice(&little_endian_limbs(affine.z().value()));
    }
    out
}

fn gpu_result_to_point<C>(limbs: &[u64]) -> ShortWeierstrassProjectivePoint<C>
where
    C: IsShortWeierstrass,
    C::BaseField: IsField<BaseType = Scalar>,
{
    assert_eq!(limbs.len(), 12);
    let x = FieldElement::<C::BaseField>::from_raw(unsigned_from_little_endian_limbs(&limbs[0..4]));
    let y = FieldElement::<C::BaseField>::from_raw(unsigned_from_little_endian_limbs(&limbs[4..8]));
    let z =
        FieldElement::<C::BaseField>::from_raw(unsigned_from_little_endian_limbs(&limbs[8..12]));

    if z == FieldElement::zero() {
        return ShortWeierstrassProjectivePoint::<C>::neutral_element();
    }

    let projective_x = x * z.clone();
    let projective_z = z.pow(3_u16);
    ShortWeierstrassProjectivePoint::<C>::new_unchecked([projective_x, y, projective_z])
}

fn mina_pallas_generator() -> LwPallasPoint {
    let x = FieldElement::from(&Scalar::from_u64(1));
    let y =
        FieldElement::from_hex("1b74b5a30a12937c53dfa9f06378ee548f655bd4333d477119cf7a23caed2abb")
            .expect("valid Mina Pallas generator y");
    PallasCurve::create_point_from_affine(x, y).expect("valid Mina Pallas generator")
}

fn mina_vesta_generator() -> LwVestaPoint {
    let x = FieldElement::from(&Scalar::from_u64(1));
    let y =
        FieldElement::from_hex("1943666ea922ae6b13b64e3aae89754cacce3a7f298ba20c4e4389b9b0276a62")
            .expect("valid Mina Vesta generator y");
    VestaCurve::create_point_from_affine(x, y).expect("valid Mina Vesta generator")
}

fn unsigned_from_bigint(value: BigInteger256) -> Scalar {
    let limbs = value.0;
    Scalar::from_limbs([limbs[3], limbs[2], limbs[1], limbs[0]])
}

fn little_endian_limbs(value: &Scalar) -> [u64; 4] {
    [
        value.limbs[3],
        value.limbs[2],
        value.limbs[1],
        value.limbs[0],
    ]
}

fn unsigned_from_little_endian_limbs(limbs: &[u64]) -> Scalar {
    assert_eq!(limbs.len(), 4);
    Scalar::from_limbs([limbs[3], limbs[2], limbs[1], limbs[0]])
}

fn field_eq<F, LwF>(ark: F, lw: &FieldElement<LwF>) -> bool
where
    F: PrimeField<BigInt = BigInteger256>,
    LwF: lambdaworks_math::field::traits::IsPrimeField<CanonicalType = Scalar>,
{
    unsigned_from_bigint(ark.into_bigint()) == lw.canonical()
}

fn bench_pallas(c: &mut Criterion) {
    bench_curve::<mina_curves::pasta::Pallas, PallasCurve>(
        c,
        "pallas-pippenger-msm",
        pallas_data,
        PippengerMSMConfig::pallas,
    );
}

fn bench_vesta(c: &mut Criterion) {
    bench_curve::<mina_curves::pasta::Vesta, VestaCurve>(
        c,
        "vesta-pippenger-msm",
        vesta_data,
        PippengerMSMConfig::vesta,
    );
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_pallas, bench_vesta
}
criterion_main!(benches);
