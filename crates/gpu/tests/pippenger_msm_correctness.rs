#![cfg(feature = "metal")]

use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{BigInteger256, PrimeField, UniformRand};
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
const WINDOW_SIZE: usize = 8;
const TEST_SIZES: &[usize] = &[1, 2, 8, 32];

type Scalar = UnsignedInteger<4>;
type LwPallasPoint = ShortWeierstrassProjectivePoint<PallasCurve>;
type LwVestaPoint = ShortWeierstrassProjectivePoint<VestaCurve>;

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
        lw_points.push(lw_generator.operate_with_self(unsigned_from_bigint(point_scalar_bigint)));
    }

    MsmData {
        ark_points,
        ark_scalars,
        lw_scalars,
        lw_points,
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

fn assert_arkworks_lambdaworks_match<ArkAffine, LwC>(data: &MsmData<ArkAffine, LwC>)
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
    let lw = pippenger::msm_with_signed(&data.lw_scalars, &data.lw_points, WINDOW_SIZE).to_affine();

    assert!(
        field_eq(ark.x().expect("non-infinity Arkworks MSM result"), lw.x()),
        "MSM x-coordinate mismatch"
    );
    assert!(
        field_eq(ark.y().expect("non-infinity Arkworks MSM result"), lw.y()),
        "MSM y-coordinate mismatch"
    );
}

fn assert_lambdaworks_cpu_gpu_match<ArkAffine, LwC>(
    config: PippengerMSMConfig,
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
    let gpu_scalars = gpu_scalars(&data.lw_scalars);
    let gpu_points = gpu_points(&data.lw_points);

    let mut msm = MetalPippengerMSM::new(config).expect("Metal device required");
    let actual_limbs = msm
        .compute(&gpu_scalars, &gpu_points)
        .expect("Metal Pippenger MSM failed");
    let actual = gpu_result_to_point::<LwC>(&actual_limbs).to_affine();

    assert_eq!(actual, expected, "CPU/GPU MSM mismatch");
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

#[test]
fn pallas_arkworks_and_lambdaworks_match() {
    for &size in TEST_SIZES {
        assert_arkworks_lambdaworks_match(&pallas_data(size));
    }
}

#[test]
fn vesta_arkworks_and_lambdaworks_match() {
    for &size in TEST_SIZES {
        assert_arkworks_lambdaworks_match(&vesta_data(size));
    }
}

#[test]
fn pallas_lambdaworks_cpu_and_metal_pippenger_match() {
    for &size in TEST_SIZES {
        let mut config = PippengerMSMConfig::pallas();
        config.window_size = WINDOW_SIZE;
        config.chunk_size = 16;
        assert_lambdaworks_cpu_gpu_match(config, &pallas_data(size));
    }
}

#[test]
fn vesta_lambdaworks_cpu_and_metal_pippenger_match() {
    for &size in TEST_SIZES {
        let mut config = PippengerMSMConfig::vesta();
        config.window_size = WINDOW_SIZE;
        config.chunk_size = 16;
        assert_lambdaworks_cpu_gpu_match(config, &vesta_data(size));
    }
}
