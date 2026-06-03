//! Benchmarks against real o1js/Kimchi MSM inputs.
//!
//! Uses the 30 datasets from `data/kimchi-internal-msm.json` captured during
//! an actual Kimchi proving run (15 Vesta × 2 048 pts, 15 Pallas × 8 192 pts).
//! Each group runs: cpu-parallel, hip-pippenger (ROCm) and metal-pippenger.
//!
//! Run:
//!   cargo bench -p lambdaworks-gpu --features rocm  --bench o1js_msm
//!   cargo bench -p lambdaworks-gpu --features metal --bench o1js_msm

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lambdaworks_math::{
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
use num_bigint::BigUint;
use serde::Deserialize;
use std::path::PathBuf;
use std::time::Duration;

#[cfg(feature = "metal")]
use lambdaworks_gpu::metal::pippenger_msm::MetalPippengerMSM;

#[cfg(feature = "rocm")]
use lambdaworks_gpu::rocm::pippenger_msm::HipPippengerMSM;

type Scalar = UnsignedInteger<4>;

// ─── JSON schema ─────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct Root {
    datasets: Vec<Dataset>,
}

#[derive(Deserialize)]
struct Dataset {
    curve: String,
    scalars: Vec<String>,
    points: Vec<Point>,
}

#[derive(Deserialize)]
struct Point {
    x: String,
    y: String,
}

// ─── pre-encoded bench payload ────────────────────────────────────────────────

struct BenchSet {
    curve: String,
    n: usize,
    lw_scalars: Vec<Scalar>,
    lw_points_pallas: Vec<ShortWeierstrassProjectivePoint<PallasCurve>>,
    lw_points_vesta: Vec<ShortWeierstrassProjectivePoint<VestaCurve>>,
    gpu_scalars: Vec<u64>,
    gpu_points: Vec<u64>,
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn data_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("data/kimchi-internal-msm.json")
}

fn decimal_to_scalar(s: &str) -> Scalar {
    let big = s.parse::<BigUint>().expect("invalid scalar");
    let bytes = big.to_bytes_be();
    let mut buf = [0u8; 32];
    let off = 32usize.saturating_sub(bytes.len());
    buf[off..].copy_from_slice(&bytes[..bytes.len().min(32)]);
    let l0 = u64::from_be_bytes(buf[0..8].try_into().unwrap());
    let l1 = u64::from_be_bytes(buf[8..16].try_into().unwrap());
    let l2 = u64::from_be_bytes(buf[16..24].try_into().unwrap());
    let l3 = u64::from_be_bytes(buf[24..32].try_into().unwrap());
    Scalar::from_limbs([l0, l1, l2, l3])
}

fn decimal_to_fe<C: IsShortWeierstrass>(s: &str) -> FieldElement<C::BaseField>
where
    C::BaseField: IsField<BaseType = Scalar> + IsPrimeField<CanonicalType = Scalar>,
{
    let big = s.parse::<BigUint>().expect("invalid field element");
    FieldElement::<C::BaseField>::from_hex(&format!("{:064x}", big)).expect("from hex")
}

fn le_limbs(v: &Scalar) -> [u64; 4] {
    [v.limbs[3], v.limbs[2], v.limbs[1], v.limbs[0]]
}

fn encode_scalars(scalars: &[Scalar]) -> Vec<u64> {
    scalars.iter().flat_map(|s| le_limbs(s)).collect()
}

fn encode_points_pallas(points: &[ShortWeierstrassProjectivePoint<PallasCurve>]) -> Vec<u64> {
    let mut out = Vec::with_capacity(points.len() * 12);
    for p in points {
        let a = p.to_affine();
        out.extend_from_slice(&le_limbs(a.x().value()));
        out.extend_from_slice(&le_limbs(a.y().value()));
        out.extend_from_slice(&le_limbs(a.z().value()));
    }
    out
}

fn encode_points_vesta(points: &[ShortWeierstrassProjectivePoint<VestaCurve>]) -> Vec<u64> {
    let mut out = Vec::with_capacity(points.len() * 12);
    for p in points {
        let a = p.to_affine();
        out.extend_from_slice(&le_limbs(a.x().value()));
        out.extend_from_slice(&le_limbs(a.y().value()));
        out.extend_from_slice(&le_limbs(a.z().value()));
    }
    out
}

fn window(n: usize) -> usize {
    match n {
        0..=4096 => 7,
        _ => 9,
    }
}

fn load_bench_sets() -> Vec<BenchSet> {
    let bytes = std::fs::read(data_path()).expect("data/kimchi-internal-msm.json not found");
    let root: Root = serde_json::from_slice(&bytes).expect("invalid JSON");

    root.datasets
        .into_iter()
        .map(|ds| {
            let scalars: Vec<Scalar> = ds.scalars.iter().map(|s| decimal_to_scalar(s)).collect();
            let n = scalars.len();
            let gpu_s = encode_scalars(&scalars);

            match ds.curve.as_str() {
                "pallas" => {
                    let points: Vec<ShortWeierstrassProjectivePoint<PallasCurve>> = ds
                        .points
                        .iter()
                        .map(|p| {
                            let x = decimal_to_fe::<PallasCurve>(&p.x);
                            let y = decimal_to_fe::<PallasCurve>(&p.y);
                            PallasCurve::create_point_from_affine(x, y).unwrap()
                        })
                        .collect();
                    let gpu_p = encode_points_pallas(&points);
                    BenchSet {
                        curve: "pallas".into(),
                        n,
                        lw_scalars: scalars,
                        lw_points_pallas: points,
                        lw_points_vesta: vec![],
                        gpu_scalars: gpu_s,
                        gpu_points: gpu_p,
                    }
                }
                "vesta" => {
                    let points: Vec<ShortWeierstrassProjectivePoint<VestaCurve>> = ds
                        .points
                        .iter()
                        .map(|p| {
                            let x = decimal_to_fe::<VestaCurve>(&p.x);
                            let y = decimal_to_fe::<VestaCurve>(&p.y);
                            VestaCurve::create_point_from_affine(x, y).unwrap()
                        })
                        .collect();
                    let gpu_p = encode_points_vesta(&points);
                    BenchSet {
                        curve: "vesta".into(),
                        n,
                        lw_scalars: scalars,
                        lw_points_pallas: vec![],
                        lw_points_vesta: points,
                        gpu_scalars: gpu_s,
                        gpu_points: gpu_p,
                    }
                }
                c => panic!("unknown curve: {}", c),
            }
        })
        .collect()
}

// ─── benchmark ───────────────────────────────────────────────────────────────

fn bench_o1js(c: &mut Criterion) {
    let sets = load_bench_sets();

    // Group by (curve, n) and bench the first dataset of each combination.
    // All datasets with the same (curve, n) use the same generators so one
    // representative suffices for performance measurement.
    let mut seen = std::collections::HashSet::new();
    for set in &sets {
        let key = (set.curve.clone(), set.n);
        if !seen.insert(key.clone()) {
            continue;
        }

        let label = format!("{}-{}", set.curve, set.n);
        let w = window(set.n);

        let mut group = c.benchmark_group(format!("o1js-msm/{}", label));

        // CPU parallel
        match set.curve.as_str() {
            "pallas" => {
                group.bench_function("cpu-parallel", |b| {
                    b.iter(|| {
                        black_box(pippenger::parallel_msm_with_signed(
                            &set.lw_scalars,
                            &set.lw_points_pallas,
                            w,
                        ))
                    })
                });
            }
            "vesta" => {
                group.bench_function("cpu-parallel", |b| {
                    b.iter(|| {
                        black_box(pippenger::parallel_msm_with_signed(
                            &set.lw_scalars,
                            &set.lw_points_vesta,
                            w,
                        ))
                    })
                });
            }
            _ => {}
        }

        // ROCm/HIP
        #[cfg(feature = "rocm")]
        {
            let hip_msm_result = match set.curve.as_str() {
                "pallas" => HipPippengerMSM::new_pallas(),
                "vesta" => HipPippengerMSM::new_vesta(),
                _ => unreachable!(),
            };
            if let Ok(base) = hip_msm_result {
                let config = base.config_for_num_points(set.n);
                if let Ok(mut msm) = HipPippengerMSM::new(config) {
                    if let Ok(prepared) = msm.prepare(&set.gpu_scalars, &set.gpu_points) {
                        group.bench_function("hip-pippenger", |b| {
                            b.iter(|| {
                                black_box(msm.compute_prepared(&prepared).expect("HIP MSM failed"))
                            })
                        });
                    }
                }
            }
        }

        // Metal
        #[cfg(feature = "metal")]
        {
            let metal_msm_result = match set.curve.as_str() {
                "pallas" => MetalPippengerMSM::new_pallas(),
                "vesta" => MetalPippengerMSM::new_vesta(),
                _ => unreachable!(),
            };
            if let Ok(base) = metal_msm_result {
                let config = base.config_for_num_points(set.n);
                if let Ok(mut msm) = MetalPippengerMSM::new(config) {
                    if let Ok(prepared) = msm.prepare(&set.gpu_scalars, &set.gpu_points) {
                        group.bench_function("metal-pippenger", |b| {
                            b.iter(|| {
                                black_box(
                                    msm.compute_prepared(&prepared).expect("Metal MSM failed"),
                                )
                            })
                        });
                    }
                }
            }
        }

        group.finish();
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(10));
    targets = bench_o1js
}
criterion_main!(benches);
