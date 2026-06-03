//! Single-run timing over all o1js/Kimchi MSM datasets.
//!
//! Loads `data/kimchi-internal-msm.json` (30 datasets: 15 Vesta×2048, 15 Pallas×8192),
//! runs CPU and GPU implementations concurrently, prints per-dataset and total timings.
//!
//! CPU implementations (cpu-parallel, arkworks) run in parallel threads.
//! GPU runs on the main thread concurrently with CPU.
//!
//! Run:
//!   cargo bench -p lambdaworks-gpu --features rocm  --bench o1js_msm
//!   cargo bench -p lambdaworks-gpu --features metal --bench o1js_msm

use ark_ec::{AffineRepr, VariableBaseMSM};
use ark_ff::{BigInteger256, PrimeField};
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
use std::{path::PathBuf, sync::Arc, time::{Duration, Instant}};

#[cfg(feature = "metal")]
use lambdaworks_gpu::metal::pippenger_msm::MetalPippengerMSM;

#[cfg(feature = "rocm")]
use lambdaworks_gpu::rocm::pippenger_msm::HipPippengerMSM;

type Scalar = UnsignedInteger<4>;
type ArkPallasAffine = mina_curves::pasta::Pallas;
type ArkVestaAffine = mina_curves::pasta::Vesta;
type ArkPallasScalar = mina_curves::pasta::Fq;
type ArkVestaScalar = mina_curves::pasta::Fp;

// ─── JSON schema ─────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct Root {
    datasets: Vec<Dataset>,
}

#[derive(Deserialize)]
struct Dataset {
    curve: String,
    scalars: Vec<String>,
    points: Vec<JsonPoint>,
}

#[derive(Deserialize)]
struct JsonPoint {
    x: String,
    y: String,
}

// ─── prepared dataset ─────────────────────────────────────────────────────────

struct PreparedDataset {
    curve: String,
    n: usize,
    // lambdaworks
    lw_scalars: Vec<Scalar>,
    lw_points_pallas: Vec<ShortWeierstrassProjectivePoint<PallasCurve>>,
    lw_points_vesta: Vec<ShortWeierstrassProjectivePoint<VestaCurve>>,
    // arkworks
    ark_pallas_points: Vec<ArkPallasAffine>,
    ark_pallas_scalars: Vec<ArkPallasScalar>,
    ark_vesta_points: Vec<ArkVestaAffine>,
    ark_vesta_scalars: Vec<ArkVestaScalar>,
    // GPU (flat u64 buffers)
    gpu_scalars: Vec<u64>,
    gpu_points: Vec<u64>,
}

// ─── conversion helpers ───────────────────────────────────────────────────────

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

fn decimal_to_lw_fe<C: IsShortWeierstrass>(s: &str) -> FieldElement<C::BaseField>
where
    C::BaseField: IsField<BaseType = Scalar> + IsPrimeField<CanonicalType = Scalar>,
{
    let big = s.parse::<BigUint>().expect("invalid field element");
    FieldElement::<C::BaseField>::from_hex(&format!("{:064x}", big)).expect("from hex")
}

fn decimal_to_ark_fe<F: PrimeField<BigInt = BigInteger256>>(s: &str) -> F {
    let big = s.parse::<BigUint>().expect("invalid field element");
    let bytes = big.to_bytes_be();
    let mut buf = [0u8; 32];
    let off = 32usize.saturating_sub(bytes.len());
    buf[off..].copy_from_slice(&bytes[..bytes.len().min(32)]);
    // BigInteger256 stores limbs LE: [limb0=LSB, ..., limb3=MSB]
    let l0 = u64::from_be_bytes(buf[0..8].try_into().unwrap());
    let l1 = u64::from_be_bytes(buf[8..16].try_into().unwrap());
    let l2 = u64::from_be_bytes(buf[16..24].try_into().unwrap());
    let l3 = u64::from_be_bytes(buf[24..32].try_into().unwrap());
    F::from_bigint(BigInteger256::new([l3, l2, l1, l0])).expect("field element out of range")
}

fn le_limbs(v: &Scalar) -> [u64; 4] {
    [v.limbs[3], v.limbs[2], v.limbs[1], v.limbs[0]]
}

fn encode_scalars(scalars: &[Scalar]) -> Vec<u64> {
    scalars.iter().flat_map(|s| le_limbs(s)).collect()
}

fn encode_points_pallas(points: &[ShortWeierstrassProjectivePoint<PallasCurve>]) -> Vec<u64> {
    let mut out = Vec::with_capacity(points.len() * 8);
    for p in points {
        let a = p.to_affine();
        out.extend_from_slice(&le_limbs(a.x().value()));
        out.extend_from_slice(&le_limbs(a.y().value()));
    }
    out
}

fn encode_points_vesta(points: &[ShortWeierstrassProjectivePoint<VestaCurve>]) -> Vec<u64> {
    let mut out = Vec::with_capacity(points.len() * 8);
    for p in points {
        let a = p.to_affine();
        out.extend_from_slice(&le_limbs(a.x().value()));
        out.extend_from_slice(&le_limbs(a.y().value()));
    }
    out
}

fn window(n: usize) -> usize {
    match n {
        0..=4 => 2,
        5..=32 => 4,
        33..=256 => 6,
        _ => 7,
    }
}

// ─── dataset loading ──────────────────────────────────────────────────────────

fn data_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .parent().unwrap()
        .join("data/kimchi-internal-msm.json")
}

fn load_datasets() -> Vec<PreparedDataset> {
    let path = data_path();
    let bytes = std::fs::read(&path).unwrap_or_else(|_| {
        panic!(
            "data file not found: {}\nExtract it with: unzip kimchi-internal-msm.zip -d data/",
            path.display()
        )
    });
    let root: Root = serde_json::from_slice(&bytes).expect("invalid JSON");

    root.datasets.into_iter().map(|ds| {
        let lw_scalars: Vec<Scalar> = ds.scalars.iter().map(|s| decimal_to_scalar(s)).collect();
        let n = lw_scalars.len();
        let gpu_scalars = encode_scalars(&lw_scalars);

        match ds.curve.as_str() {
            "pallas" => {
                let lw_points: Vec<ShortWeierstrassProjectivePoint<PallasCurve>> = ds.points.iter()
                    .map(|p| {
                        let x = decimal_to_lw_fe::<PallasCurve>(&p.x);
                        let y = decimal_to_lw_fe::<PallasCurve>(&p.y);
                        PallasCurve::create_point_from_affine(x, y).unwrap()
                    })
                    .collect();
                let ark_points: Vec<ArkPallasAffine> = ds.points.iter()
                    .map(|p| {
                        let x = decimal_to_ark_fe::<
                            <ArkPallasAffine as AffineRepr>::BaseField>(&p.x);
                        let y = decimal_to_ark_fe::<
                            <ArkPallasAffine as AffineRepr>::BaseField>(&p.y);
                        ArkPallasAffine::new(x, y)
                    })
                    .collect();
                let ark_scalars: Vec<ArkPallasScalar> = ds.scalars.iter()
                    .map(|s| decimal_to_ark_fe(s))
                    .collect();
                let gpu_points = encode_points_pallas(&lw_points);
                PreparedDataset {
                    curve: "pallas".into(), n,
                    lw_scalars, lw_points_pallas: lw_points, lw_points_vesta: vec![],
                    ark_pallas_points: ark_points, ark_pallas_scalars: ark_scalars,
                    ark_vesta_points: vec![], ark_vesta_scalars: vec![],
                    gpu_scalars, gpu_points,
                }
            }
            "vesta" => {
                let lw_points: Vec<ShortWeierstrassProjectivePoint<VestaCurve>> = ds.points.iter()
                    .map(|p| {
                        let x = decimal_to_lw_fe::<VestaCurve>(&p.x);
                        let y = decimal_to_lw_fe::<VestaCurve>(&p.y);
                        VestaCurve::create_point_from_affine(x, y).unwrap()
                    })
                    .collect();
                let ark_points: Vec<ArkVestaAffine> = ds.points.iter()
                    .map(|p| {
                        let x = decimal_to_ark_fe::<
                            <ArkVestaAffine as AffineRepr>::BaseField>(&p.x);
                        let y = decimal_to_ark_fe::<
                            <ArkVestaAffine as AffineRepr>::BaseField>(&p.y);
                        ArkVestaAffine::new(x, y)
                    })
                    .collect();
                let ark_scalars: Vec<ArkVestaScalar> = ds.scalars.iter()
                    .map(|s| decimal_to_ark_fe(s))
                    .collect();
                let gpu_points = encode_points_vesta(&lw_points);
                PreparedDataset {
                    curve: "vesta".into(), n,
                    lw_scalars, lw_points_pallas: vec![], lw_points_vesta: lw_points,
                    ark_pallas_points: vec![], ark_pallas_scalars: vec![],
                    ark_vesta_points: ark_points, ark_vesta_scalars: ark_scalars,
                    gpu_scalars, gpu_points,
                }
            }
            c => panic!("unknown curve: {c}"),
        }
    }).collect()
}

// ─── display ─────────────────────────────────────────────────────────────────

fn fmt_ms(d: Duration) -> String {
    format!("{:.1} ms", d.as_secs_f64() * 1000.0)
}

fn print_summary(label: &str, timings: &[Duration]) {
    let total: Duration = timings.iter().sum();
    let avg = total / timings.len() as u32;
    let mut sorted = timings.to_vec();
    sorted.sort();
    let median = sorted[sorted.len() / 2];
    println!(
        "  {:<30} total={:>10}  avg={:>10}  median={:>10}",
        label, fmt_ms(total), fmt_ms(avg), fmt_ms(median),
    );
}

// ─── runners ─────────────────────────────────────────────────────────────────

fn run_cpu_parallel(datasets: &[PreparedDataset]) -> Vec<Duration> {
    datasets.iter().map(|ds| {
        let w = window(ds.n);
        let t = Instant::now();
        match ds.curve.as_str() {
            "pallas" => { let _ = pippenger::parallel_msm_with_signed(&ds.lw_scalars, &ds.lw_points_pallas, w); }
            "vesta"  => { let _ = pippenger::parallel_msm_with_signed(&ds.lw_scalars, &ds.lw_points_vesta, w); }
            _ => unreachable!(),
        }
        t.elapsed()
    }).collect()
}

fn run_arkworks(datasets: &[PreparedDataset]) -> Vec<Duration> {
    datasets.iter().map(|ds| {
        let t = Instant::now();
        match ds.curve.as_str() {
            "pallas" => {
                let _ = <mina_curves::pasta::ProjectivePallas as VariableBaseMSM>::msm(
                    &ds.ark_pallas_points, &ds.ark_pallas_scalars,
                ).expect("ark pallas msm");
            }
            "vesta" => {
                let _ = <mina_curves::pasta::ProjectiveVesta as VariableBaseMSM>::msm(
                    &ds.ark_vesta_points, &ds.ark_vesta_scalars,
                ).expect("ark vesta msm");
            }
            _ => unreachable!(),
        }
        t.elapsed()
    }).collect()
}

// ─── main ─────────────────────────────────────────────────────────────────────

fn main() {
    println!("Loading and preparing datasets...");
    let datasets = Arc::new(load_datasets());
    println!("Loaded {} datasets.\n", datasets.len());

    // Spawn CPU threads — they share rayon's global pool, which is fine.
    let ds_cpu = Arc::clone(&datasets);
    let cpu_par_thread = std::thread::spawn(move || run_cpu_parallel(&ds_cpu));

    let ds_ark = Arc::clone(&datasets);
    let ark_thread = std::thread::spawn(move || run_arkworks(&ds_ark));

    // ── ROCm/HIP batched (main thread, concurrent with CPU threads) ──────────
    // Group by curve and run a single batched kernel launch per group,
    // instead of 30 sequential launches.
    #[cfg(feature = "rocm")]
    let hip_timings: Option<Vec<Duration>> = {
        let pallas_idx: Vec<usize> = datasets.iter().enumerate()
            .filter(|(_, d)| d.curve == "pallas").map(|(i, _)| i).collect();
        let vesta_idx: Vec<usize>  = datasets.iter().enumerate()
            .filter(|(_, d)| d.curve == "vesta").map(|(i, _)| i).collect();

        let mut timings = vec![Duration::ZERO; datasets.len()];
        let mut ok = true;

        for (curve_name, indices) in [("pallas", &pallas_idx), ("vesta", &vesta_idx)] {
            if indices.is_empty() { continue; }
            let n = datasets[indices[0]].n;

            let base = match curve_name {
                "pallas" => HipPippengerMSM::new_pallas(),
                _        => HipPippengerMSM::new_vesta(),
            };
            let base = match base {
                Ok(b) => b,
                Err(e) => { eprintln!("HIP init failed: {e}"); ok = false; break; }
            };
            let config = base.config_for_num_points(n);
            let mut msm = match HipPippengerMSM::new(config) {
                Ok(m) => m,
                Err(e) => { eprintln!("HIP new failed: {e}"); ok = false; break; }
            };

            let batch: Vec<(&[u64], &[u64])> = indices.iter()
                .map(|&i| (datasets[i].gpu_scalars.as_slice(), datasets[i].gpu_points.as_slice()))
                .collect();

            let t = Instant::now();
            let results = match msm.compute_batch(&batch) {
                Ok(r) => r,
                Err(e) => { eprintln!("HIP batch failed: {e}"); ok = false; break; }
            };
            let elapsed = t.elapsed();
            let per = elapsed / indices.len() as u32;
            let _ = results;

            println!("  hip {:<10} n={:<6} batch={:<3} total={} per={}",
                curve_name, n, indices.len(), fmt_ms(elapsed), fmt_ms(per));

            for &i in indices { timings[i] = per; }
        }

        if ok { Some(timings) } else { None }
    };

    // ── Metal (main thread) ───────────────────────────────────────────────────
    #[cfg(feature = "metal")]
    let metal_timings: Option<Vec<Duration>> = {
        let mut timings = Vec::with_capacity(datasets.len());
        let mut ok = true;
        for ds in datasets.iter() {
            let msm_result = match ds.curve.as_str() {
                "pallas" => MetalPippengerMSM::new_pallas(),
                "vesta"  => MetalPippengerMSM::new_vesta(),
                _ => unreachable!(),
            };
            let base = match msm_result {
                Ok(b) => b,
                Err(e) => { eprintln!("Metal init failed: {e}"); ok = false; break; }
            };
            let config = base.config_for_num_points(ds.n);
            let mut msm = match MetalPippengerMSM::new(config) {
                Ok(m) => m,
                Err(e) => { eprintln!("Metal new failed: {e}"); ok = false; break; }
            };
            let t = Instant::now();
            let _ = msm.compute(&ds.gpu_scalars, &ds.gpu_points).expect("Metal compute");
            timings.push(t.elapsed());
        }
        if ok && !timings.is_empty() { Some(timings) } else { None }
    };

    // ── collect CPU results ───────────────────────────────────────────────────
    let cpu_par_timings = cpu_par_thread.join().expect("cpu-parallel thread panicked");
    let ark_timings = ark_thread.join().expect("arkworks thread panicked");

    // ── print results ─────────────────────────────────────────────────────────

    // Per-dataset table
    println!("{:<6} {:<10} {:<8} {:>14} {:>14} {:>14} {:>14}",
        "idx", "curve", "n",
        "cpu-parallel", "arkworks",
        if cfg!(feature = "rocm") { "hip" } else if cfg!(feature = "metal") { "metal" } else { "" },
        "",
    );
    println!("{}", "-".repeat(80));
    for (i, ds) in datasets.iter().enumerate() {
        let cpu_ms  = fmt_ms(cpu_par_timings[i]);
        let ark_ms  = fmt_ms(ark_timings[i]);
        #[cfg(feature = "rocm")]
        let gpu_ms = hip_timings.as_ref().map_or("—".into(), |t| fmt_ms(t[i]));
        #[cfg(feature = "metal")]
        let gpu_ms = metal_timings.as_ref().map_or("—".into(), |t| fmt_ms(t[i]));
        #[cfg(not(any(feature = "rocm", feature = "metal")))]
        let gpu_ms = String::from("—");
        println!("{:<6} {:<10} {:<8} {:>14} {:>14} {:>14}",
            i, ds.curve, ds.n, cpu_ms, ark_ms, gpu_ms);
    }

    println!();
    println!("=== Summary ===");
    print_summary("cpu-parallel", &cpu_par_timings);
    print_summary("arkworks", &ark_timings);
    #[cfg(feature = "rocm")]
    if let Some(ref t) = hip_timings { print_summary("hip-pippenger", t); }
    #[cfg(feature = "metal")]
    if let Some(ref t) = metal_timings { print_summary("metal-pippenger", t); }
}
