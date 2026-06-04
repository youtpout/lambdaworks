//! Correctness tests for the ROCm/HIP Pippenger MSM.
//!
//! Each GPU result is compared against:
//!   1. The CPU signed-digit Pippenger (lambdaworks-math).
//!   2. The Arkworks `VariableBaseMSM` reference implementation.

#![cfg(feature = "rocm")]

use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{BigInteger256, PrimeField, UniformRand};
use lambdaworks_gpu::rocm::pippenger_msm::{HipPippengerMSM, HipPippengerMSMConfig};
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
const WINDOW_SIZE: usize = 4; // capped by MAX_PRIVATE_BUCKETS kernel limit
const TEST_SIZES: &[usize] = &[2, 8, 32];

type Scalar = UnsignedInteger<4>;
type LwPallasPoint = ShortWeierstrassProjectivePoint<PallasCurve>;
type LwVestaPoint = ShortWeierstrassProjectivePoint<VestaCurve>;

// ─── data helpers ─────────────────────────────────────────────────────────────

struct MsmData<ArkAffine, LwC>
where
    ArkAffine: AffineRepr,
    ArkAffine::ScalarField: PrimeField<BigInt = BigInteger256>,
    LwC: IsShortWeierstrass,
{
    ark_points: Vec<ArkAffine>,
    ark_scalars: Vec<ArkAffine::ScalarField>,
    lw_scalars: Vec<Scalar>,
    lw_points: Vec<ShortWeierstrassProjectivePoint<LwC>>,
}

fn pallas_data(size: usize) -> MsmData<mina_curves::pasta::Pallas, PallasCurve> {
    make_data::<mina_curves::pasta::ProjectivePallas, PallasCurve>(size, mina_pallas_generator())
}

fn vesta_data(size: usize) -> MsmData<mina_curves::pasta::Vesta, VestaCurve> {
    make_data::<mina_curves::pasta::ProjectiveVesta, VestaCurve>(size, mina_vesta_generator())
}

fn make_data<ArkG, LwC>(
    size: usize,
    lw_generator: ShortWeierstrassProjectivePoint<LwC>,
) -> MsmData<ArkG::Affine, LwC>
where
    ArkG: CurveGroup,
    ArkG::Affine: AffineRepr<Group = ArkG>,
    ArkG::ScalarField: PrimeField<BigInt = BigInteger256> + UniformRand,
    LwC: IsShortWeierstrass
        + IsEllipticCurve<PointRepresentation = ShortWeierstrassProjectivePoint<LwC>>,
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
        lw_points.push(
            lw_generator.operate_with_self(unsigned_from_bigint(point_scalar_bigint)),
        );
    }

    MsmData {
        ark_points,
        ark_scalars,
        lw_scalars,
        lw_points,
    }
}

// ─── assertion helpers ────────────────────────────────────────────────────────

fn ark_msm<G>(points: &[G], scalars: &[G::ScalarField]) -> G::Group
where
    G: AffineRepr,
    G::Group: VariableBaseMSM<MulBase = G>,
    G::ScalarField: PrimeField<BigInt = BigInteger256>,
{
    <G::Group as VariableBaseMSM>::msm(points, scalars).expect("valid Arkworks MSM input")
}

fn assert_ark_lw_match<ArkAffine, LwC>(data: &MsmData<ArkAffine, LwC>)
where
    ArkAffine: AffineRepr,
    ArkAffine::BaseField: PrimeField<BigInt = BigInteger256>,
    ArkAffine::Group: VariableBaseMSM<MulBase = ArkAffine>,
    ArkAffine::ScalarField: PrimeField<BigInt = BigInteger256>,
    LwC: IsShortWeierstrass
        + IsEllipticCurve<PointRepresentation = ShortWeierstrassProjectivePoint<LwC>>,
    LwC::BaseField: IsField<BaseType = Scalar> + IsPrimeField<CanonicalType = Scalar>,
{
    let ark = ark_msm(&data.ark_points, &data.ark_scalars).into_affine();
    let lw =
        pippenger::msm_with_signed(&data.lw_scalars, &data.lw_points, WINDOW_SIZE).to_affine();
    assert!(field_eq(ark.x().expect("non-infinity"), lw.x()), "x mismatch (ark vs lw)");
    assert!(field_eq(ark.y().expect("non-infinity"), lw.y()), "y mismatch (ark vs lw)");
}

fn assert_cpu_gpu_match<ArkAffine, LwC>(
    config: HipPippengerMSMConfig,
    data: &MsmData<ArkAffine, LwC>,
) where
    ArkAffine: AffineRepr,
    ArkAffine::ScalarField: PrimeField<BigInt = BigInteger256>,
    LwC: IsShortWeierstrass
        + IsEllipticCurve<PointRepresentation = ShortWeierstrassProjectivePoint<LwC>>,
    LwC::BaseField: IsField<BaseType = Scalar>,
{
    let expected =
        pippenger::msm_with_signed(&data.lw_scalars, &data.lw_points, WINDOW_SIZE).to_affine();
    let gpu_scalars = encode_scalars(&data.lw_scalars);
    let gpu_points = encode_points(&data.lw_points);

    let mut msm = HipPippengerMSM::new(config).expect("ROCm device required");
    let actual_limbs = msm
        .compute(&gpu_scalars, &gpu_points)
        .expect("HIP Pippenger MSM failed");
    let actual = limbs_to_point::<LwC>(&actual_limbs).to_affine();

    assert_eq!(actual, expected, "CPU/GPU MSM result mismatch");
}

// ─── encoding helpers ─────────────────────────────────────────────────────────

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
    let mut out = Vec::with_capacity(points.len() * 8);
    for p in points {
        let affine = p.to_affine();
        out.extend_from_slice(&little_endian_limbs(affine.x().value()));
        out.extend_from_slice(&little_endian_limbs(affine.y().value()));
        // z is implicit (mont_one) — not sent to the GPU
    }
    out
}

fn limbs_to_point<C>(limbs: &[u64]) -> ShortWeierstrassProjectivePoint<C>
where
    C: IsShortWeierstrass,
    C::BaseField: IsField<BaseType = Scalar>,
{
    assert_eq!(limbs.len(), 12);
    let x = FieldElement::<C::BaseField>::from_raw(from_le_limbs(&limbs[0..4]));
    let y = FieldElement::<C::BaseField>::from_raw(from_le_limbs(&limbs[4..8]));
    let z = FieldElement::<C::BaseField>::from_raw(from_le_limbs(&limbs[8..12]));
    if z == FieldElement::zero() {
        return ShortWeierstrassProjectivePoint::<C>::neutral_element();
    }
    let projective_x = x * z.clone();
    let projective_z = z.pow(3_u16);
    ShortWeierstrassProjectivePoint::<C>::new_unchecked([projective_x, y, projective_z])
}

// ─── curve generators ─────────────────────────────────────────────────────────

fn mina_pallas_generator() -> LwPallasPoint {
    let x = FieldElement::from(&Scalar::from_u64(1));
    let y = FieldElement::from_hex(
        "1b74b5a30a12937c53dfa9f06378ee548f655bd4333d477119cf7a23caed2abb",
    )
    .expect("valid Mina Pallas generator y");
    PallasCurve::create_point_from_affine(x, y).expect("valid Mina Pallas generator")
}

fn mina_vesta_generator() -> LwVestaPoint {
    let x = FieldElement::from(&Scalar::from_u64(1));
    let y = FieldElement::from_hex(
        "1943666ea922ae6b13b64e3aae89754cacce3a7f298ba20c4e4389b9b0276a62",
    )
    .expect("valid Mina Vesta generator y");
    VestaCurve::create_point_from_affine(x, y).expect("valid Mina Vesta generator")
}

// ─── conversion utilities ─────────────────────────────────────────────────────

fn unsigned_from_bigint(v: BigInteger256) -> Scalar {
    Scalar::from_limbs([v.0[3], v.0[2], v.0[1], v.0[0]])
}

fn little_endian_limbs(v: &Scalar) -> [u64; 4] {
    [v.limbs[3], v.limbs[2], v.limbs[1], v.limbs[0]]
}

fn from_le_limbs(limbs: &[u64]) -> Scalar {
    assert_eq!(limbs.len(), 4);
    Scalar::from_limbs([limbs[3], limbs[2], limbs[1], limbs[0]])
}

fn field_eq<F, LwF>(ark: F, lw: &FieldElement<LwF>) -> bool
where
    F: PrimeField<BigInt = BigInteger256>,
    LwF: IsPrimeField<CanonicalType = Scalar>,
{
    unsigned_from_bigint(ark.into_bigint()) == lw.canonical()
}

// ─── test cases ───────────────────────────────────────────────────────────────

/// Sanity check: Arkworks and lambdaworks CPU MSM must agree (no GPU involved).
#[test]
fn pallas_arkworks_and_lambdaworks_match() {
    for &size in TEST_SIZES {
        assert_ark_lw_match(&pallas_data(size));
    }
}

#[test]
fn vesta_arkworks_and_lambdaworks_match() {
    for &size in TEST_SIZES {
        assert_ark_lw_match(&vesta_data(size));
    }
}

/// Core test: HIP GPU MSM must match the CPU reference.
#[test]
fn pallas_lambdaworks_cpu_and_hip_pippenger_match() {
    for &size in TEST_SIZES {
        let mut config = HipPippengerMSMConfig::pallas();
        config.window_size = WINDOW_SIZE;
        config.chunk_size = 16;
        assert_cpu_gpu_match(config, &pallas_data(size));
    }
}

#[test]
fn vesta_lambdaworks_cpu_and_hip_pippenger_match() {
    for &size in TEST_SIZES {
        let mut config = HipPippengerMSMConfig::vesta();
        config.window_size = WINDOW_SIZE;
        config.chunk_size = 16;
        assert_cpu_gpu_match(config, &vesta_data(size));
    }
}

/// Single-point MSM: result should equal the point itself (scalar = 1 in Montgomery form).
#[test]
fn single_point_identity_scalar_pallas() {
    let config = HipPippengerMSMConfig::pallas();
    // scalar = 1 (one), field element in Montgomery = R mod p
    // For simplicity, use the CPU path to generate a known point and scalar = 1 limbs.
    let gen = mina_pallas_generator();
    let one = Scalar::from_u64(1);
    let cpu_result =
        pippenger::msm_with_signed(&[one.clone()], &[gen.clone()], 4).to_affine();
    let gpu_scalars = encode_scalars(&[one]);
    let gpu_points = encode_points(&[gen]);
    let mut msm = HipPippengerMSM::new(config).expect("ROCm device required");
    let limbs = msm.compute(&gpu_scalars, &gpu_points).expect("HIP MSM failed");
    let gpu_result = limbs_to_point::<PallasCurve>(&limbs).to_affine();
    assert_eq!(gpu_result, cpu_result, "single-point Pallas MSM mismatch");
}

/// Larger batch to exercise the chunk merging path.
#[test]
fn pallas_large_batch_hip_matches_cpu() {
    let size = 512;
    let mut config = HipPippengerMSMConfig::pallas();
    config.window_size = HipPippengerMSMConfig::optimal_window_size(size);
    config.chunk_size = 64;
    assert_cpu_gpu_match(config, &pallas_data(size));
}
