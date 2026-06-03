//! ROCm/HIP Pippenger MSM for 256-bit short-Weierstrass curves.
//!
//! Mirrors `MetalPippengerMSM` in the Metal backend. Points are Jacobian
//! `(x, y, z)` with 4 little-endian `u64` limbs per coordinate stored in
//! Montgomery form. The curve model requires `a = 0` (Pallas / Vesta).

use super::abstractions::{
    errors::{HipError, HipResult},
    state::{DeviceBuffer, HipState},
};

const PIPPENGER_MSM_HIP_SOURCE: &str =
    include_str!("shaders/pippenger_msm/pippenger_msm.hip");

const COORD_LIMBS: usize = 4;
const COORDS_PER_POINT: usize = 3;
/// Limbs in a Jacobian bucket point (x,y,z — 12 u64s).
const LIMBS_PER_POINT: usize = COORD_LIMBS * COORDS_PER_POINT;
/// Limbs in an affine input point (x,y only — 8 u64s). Z is implicit (= mont_one).
const LIMBS_PER_AFFINE: usize = COORD_LIMBS * 2;

const PALLAS_MODULUS: [u64; COORD_LIMBS] = [
    0x992d30ed00000001,
    0x224698fc094cf91b,
    0x0000000000000000,
    0x4000000000000000,
];

const VESTA_MODULUS: [u64; COORD_LIMBS] = [
    0x8c46eb2100000001,
    0x224698fc0994a8dd,
    0x0000000000000000,
    0x4000000000000000,
];

// ─── public config ────────────────────────────────────────────────────────────

/// Configuration for a 256-bit Pippenger MSM over an `a = 0` curve.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HipPippengerMSMConfig {
    pub window_size: usize,
    pub scalar_limbs: usize,
    pub bits_per_limb: usize,
    pub field_modulus: [u64; COORD_LIMBS],
    pub montgomery_inv: u64,
    pub chunk_size: usize,
}

impl HipPippengerMSMConfig {
    pub fn pallas() -> Self {
        Self::for_modulus(PALLAS_MODULUS)
    }

    pub fn vesta() -> Self {
        Self::for_modulus(VESTA_MODULUS)
    }

    pub fn for_modulus(field_modulus: [u64; COORD_LIMBS]) -> Self {
        Self {
            window_size: 7,
            scalar_limbs: COORD_LIMBS,
            bits_per_limb: 64,
            montgomery_inv: montgomery_inv64(field_modulus[0]),
            field_modulus,
            chunk_size: 256,
        }
    }

    pub fn optimal_window_size(num_points: usize) -> usize {
        // Cap at 7 so num_buckets = 2^(w-1) ≤ 64, fitting in the fast register path.
        // The medium shared-memory path handles up to 512 buckets (window ≤ 10) but
        // requires per-thread bucket striping which scales poorly; register path is fastest.
        match num_points {
            0..=4 => 2,
            5..=32 => 4,
            33..=256 => 6,
            _ => 7,
        }
    }

    pub const MAX_WINDOW_SIZE: usize = 20;

    pub fn num_windows(&self) -> usize {
        self.validate();
        (self.scalar_limbs * self.bits_per_limb).div_ceil(self.window_size)
    }

    pub fn num_buckets(&self) -> usize {
        self.validate();
        1 << (self.window_size - 1)
    }

    /// Optimal chunk size for `num_points`, targeting ~4× GPU occupancy.
    ///
    /// Each MSM's partial_buckets = num_windows × num_chunks × num_buckets × 96 bytes.
    /// With window=7 (num_buckets=64) this stays well under 100 MB per MSM.
    pub fn optimal_chunk_size(
        num_points: usize,
        num_windows: usize,
        cu_count: usize,
    ) -> usize {
        // Target ~4× over-subscription relative to CU count.
        // Assume 64 threads/CU wavefront (conservative for RDNA/CDNA).
        let target_threads = cu_count * 64 * 4;
        let target_chunks = target_threads.div_ceil(num_windows);
        let raw = num_points.div_ceil(target_chunks);
        raw.max(16).next_power_of_two()
    }

    fn validate(&self) {
        assert!(
            (1..=Self::MAX_WINDOW_SIZE).contains(&self.window_size),
            "window_size must be in 1..={}",
            Self::MAX_WINDOW_SIZE
        );
        assert_eq!(self.scalar_limbs, COORD_LIMBS, "only 256-bit scalars supported");
        assert_eq!(self.bits_per_limb, 64, "only 64-bit limbs supported");
        assert!(self.chunk_size > 0, "chunk_size must be positive");
    }
}

impl Default for HipPippengerMSMConfig {
    fn default() -> Self {
        Self::pallas()
    }
}

// ─── prepared buffers ─────────────────────────────────────────────────────────

pub struct PreparedHipPippengerMSM {
    digits_buf: DeviceBuffer,
    points_buf: DeviceBuffer,
    partial_buckets_buf: DeviceBuffer,
    buckets_buf: DeviceBuffer,
    window_sums_buf: DeviceBuffer,
    accum_config_buf: DeviceBuffer,
    merge_config_buf: DeviceBuffer,
    reduction_config_buf: DeviceBuffer,
    field_buf: DeviceBuffer,
    clear_config_buf: DeviceBuffer,
    accumulation_threads: u64,
    merge_threads: u64,
    effective_windows: usize,
}

// ─── main struct ──────────────────────────────────────────────────────────────

/// ROCm/HIP Pippenger MSM.
pub struct HipPippengerMSM {
    state: HipState,
    config: HipPippengerMSMConfig,
    initialized: bool,
}

impl HipPippengerMSM {
    pub fn new(config: HipPippengerMSMConfig) -> HipResult<Self> {
        let state = HipState::new()?;
        Ok(Self {
            state,
            config,
            initialized: false,
        })
    }

    pub fn new_pallas() -> HipResult<Self> {
        Self::new(HipPippengerMSMConfig::pallas())
    }

    pub fn new_vesta() -> HipResult<Self> {
        Self::new(HipPippengerMSMConfig::vesta())
    }

    pub fn config(&self) -> &HipPippengerMSMConfig {
        &self.config
    }

    pub fn set_config(&mut self, config: HipPippengerMSMConfig) {
        self.config = config;
    }

    /// Derive an optimal config for `num_points` on this device.
    pub fn config_for_num_points(&self, num_points: usize) -> HipPippengerMSMConfig {
        let window_size = HipPippengerMSMConfig::optimal_window_size(num_points);
        let effective_windows =
            (self.config.scalar_limbs * self.config.bits_per_limb).div_ceil(window_size) + 1;
        // 60 CUs is a reasonable default for a mid-range AMD GPU (RX 6700 XT etc.)
        let chunk_size =
            HipPippengerMSMConfig::optimal_chunk_size(num_points, effective_windows, 60);
        HipPippengerMSMConfig {
            window_size,
            chunk_size,
            ..self.config.clone()
        }
    }

    pub fn initialize(&mut self) -> HipResult<()> {
        if self.initialized {
            return Ok(());
        }
        self.state
            .load_source(PIPPENGER_MSM_HIP_SOURCE, "pippenger_msm.hip")?;
        self.state.prepare_function("clear_u64_buffer")?;
        self.state
            .prepare_function("bucket_accumulation_by_chunk")?;
        self.state.prepare_function("bucket_merge")?;
        self.state.prepare_function("bucket_reduction")?;
        self.initialized = true;
        Ok(())
    }

    /// Compute `Σ scalars[i] * points[i]`.
    ///
    /// `scalars`: flat `[s0_l0, ..., s0_l3, s1_l0, ...]`.
    /// `points`:  flat `[x0_l0..x0_l3, y0_l0..y0_l3, z0_l0..z0_l3, x1_l0..]`.
    pub fn compute(&mut self, scalars: &[u64], points: &[u64]) -> HipResult<Vec<u64>> {
        let prepared = self.prepare(scalars, points)?;
        self.compute_prepared(&prepared)
    }

    pub fn prepare(
        &mut self,
        scalars: &[u64],
        points: &[u64],
    ) -> HipResult<PreparedHipPippengerMSM> {
        if !self.initialized {
            self.initialize()?;
        }

        if scalars.len() % self.config.scalar_limbs != 0 {
            return Err(HipError::InvalidInputSize {
                expected: self.config.scalar_limbs,
                actual: scalars.len(),
            });
        }
        if points.len() % LIMBS_PER_AFFINE != 0 {
            return Err(HipError::InvalidInputSize {
                expected: LIMBS_PER_AFFINE,
                actual: points.len(),
            });
        }

        let num_scalars = scalars.len() / self.config.scalar_limbs;
        let num_points = points.len() / LIMBS_PER_AFFINE;
        if num_scalars != num_points {
            return Err(HipError::LengthMismatch(num_scalars, num_points));
        }
        if num_scalars == 0 {
            return Err(HipError::EmptyInput);
        }

        let signed_digits = recode_scalars_signed(&self.config, scalars, num_scalars);
        let digits_buf = self.state.alloc_buffer_with_data(&signed_digits)?;
        let points_buf = self.state.alloc_buffer_with_data(points)?;

        let num_buckets = self.config.num_buckets();
        let effective_windows = self.config.num_windows() + 1;
        let num_chunks = num_scalars.div_ceil(self.config.chunk_size);

        let partial_len = effective_windows * num_chunks * num_buckets * LIMBS_PER_POINT;
        let partial_buckets_buf =
            self.state.alloc_buffer(partial_len * std::mem::size_of::<u64>())?;

        let buckets_len = effective_windows * num_buckets * LIMBS_PER_POINT;
        let buckets_buf =
            self.state.alloc_buffer(buckets_len * std::mem::size_of::<u64>())?;

        let window_sums_buf = self
            .state
            .alloc_buffer(effective_windows * LIMBS_PER_POINT * std::mem::size_of::<u64>())?;

        let accum_config_buf = self.state.alloc_buffer_with_data(&[
            num_scalars as u32,
            effective_windows as u32,
            num_buckets as u32,
            self.config.chunk_size as u32,
            num_chunks as u32,
        ])?;
        let merge_config_buf = self.state.alloc_buffer_with_data(&[
            effective_windows as u32,
            num_buckets as u32,
            num_chunks as u32,
        ])?;
        let reduction_config_buf = self
            .state
            .alloc_buffer_with_data(&[effective_windows as u32, num_buckets as u32])?;
        let field_buf = self.field_params_buffer()?;
        let clear_config_buf =
            self.state.alloc_buffer_with_data(&[partial_len as u32])?;

        Ok(PreparedHipPippengerMSM {
            digits_buf,
            points_buf,
            partial_buckets_buf,
            buckets_buf,
            window_sums_buf,
            accum_config_buf,
            merge_config_buf,
            reduction_config_buf,
            field_buf,
            clear_config_buf,
            accumulation_threads: (effective_windows * num_chunks) as u64,
            merge_threads: (effective_windows * num_buckets) as u64,
            effective_windows,
        })
    }

    /// Compute multiple MSMs in a single batched GPU kernel launch.
    ///
    /// All entries in `batch` must have the same number of points and use the
    /// same curve (i.e. the same `config`).  Results are returned in the same
    /// order as the input.
    pub fn compute_batch(
        &mut self,
        batch: &[(&[u64], &[u64])],
    ) -> HipResult<Vec<Vec<u64>>> {
        if batch.is_empty() {
            return Ok(vec![]);
        }
        if !self.initialized {
            self.initialize()?;
        }

        let num_msms = batch.len();

        // Validate and determine per-MSM sizes from the first entry.
        let num_scalars = batch[0].0.len() / self.config.scalar_limbs;
        let num_points  = batch[0].1.len() / LIMBS_PER_AFFINE;
        if num_scalars != num_points || num_scalars == 0 {
            return Err(HipError::LengthMismatch(num_scalars, num_points));
        }

        // Concatenate all scalars and points into single host buffers.
        let total_scalars = batch[0].0.len() * num_msms;
        let total_points  = batch[0].1.len() * num_msms;
        let mut all_scalars_raw = Vec::with_capacity(total_scalars);
        let mut all_points_raw  = Vec::with_capacity(total_points);
        for (s, p) in batch {
            all_scalars_raw.extend_from_slice(s);
            all_points_raw.extend_from_slice(p);
        }

        // Recode all scalars (flat: [msm0_digits..., msm1_digits..., ...]).
        let num_buckets     = self.config.num_buckets();
        let effective_windows = self.config.num_windows() + 1;
        let num_chunks      = num_scalars.div_ceil(self.config.chunk_size);

        let mut all_digits = Vec::with_capacity(num_msms * num_scalars * effective_windows);
        for i in 0..num_msms {
            let scalar_slice = &all_scalars_raw[i * batch[0].0.len()..][..batch[0].0.len()];
            let digits = recode_scalars_signed(&self.config, scalar_slice, num_scalars);
            all_digits.extend_from_slice(&digits);
        }

        // Upload concatenated buffers.
        let digits_buf = self.state.alloc_buffer_with_data(&all_digits)?;
        let points_buf = self.state.alloc_buffer_with_data(&all_points_raw)?;

        // Allocate batch-sized intermediate buffers.
        let partial_len = num_msms * effective_windows * num_chunks * num_buckets * LIMBS_PER_POINT;
        let partial_buckets_buf =
            self.state.alloc_buffer(partial_len * std::mem::size_of::<u64>())?;
        self.state.zero_buffer(&partial_buckets_buf)?;

        let buckets_len = num_msms * effective_windows * num_buckets * LIMBS_PER_POINT;
        let buckets_buf =
            self.state.alloc_buffer(buckets_len * std::mem::size_of::<u64>())?;

        let window_sums_buf = self.state.alloc_buffer(
            num_msms * effective_windows * LIMBS_PER_POINT * std::mem::size_of::<u64>(),
        )?;

        // Config buffers (same as single-MSM — per-MSM sizes, shared across all).
        let accum_config_buf = self.state.alloc_buffer_with_data(&[
            num_scalars as u32,
            effective_windows as u32,
            num_buckets as u32,
            self.config.chunk_size as u32,
            num_chunks as u32,
        ])?;
        let merge_config_buf = self.state.alloc_buffer_with_data(&[
            effective_windows as u32,
            num_buckets as u32,
            num_chunks as u32,
        ])?;
        let reduction_config_buf = self
            .state
            .alloc_buffer_with_data(&[effective_windows as u32, num_buckets as u32])?;
        let field_buf = self.field_params_buffer()?;

        let accum_bufs: &[&DeviceBuffer] = &[
            &digits_buf, &points_buf, &partial_buckets_buf, &accum_config_buf, &field_buf,
        ];
        let merge_bufs: &[&DeviceBuffer] = &[
            &partial_buckets_buf, &buckets_buf, &merge_config_buf, &field_buf,
        ];
        let reduc_bufs: &[&DeviceBuffer] = &[
            &buckets_buf, &window_sums_buf, &reduction_config_buf, &field_buf,
        ];

        self.state.execute_compute_seq_2d(
            &[
                ("bucket_accumulation_by_chunk", accum_bufs, (effective_windows * num_chunks) as u64),
                ("bucket_merge",                 merge_bufs, (effective_windows * num_buckets) as u64),
                ("bucket_reduction",             reduc_bufs, effective_windows as u64),
            ],
            num_msms as u32,
        )?;

        // Read back all window sums and combine on CPU.
        let all_sums: Vec<u64> = self.state.read_buffer(
            &window_sums_buf,
            num_msms * effective_windows * LIMBS_PER_POINT,
        )?;

        let results = (0..num_msms)
            .map(|i| {
                let base = i * effective_windows * LIMBS_PER_POINT;
                self.combine_windows(&all_sums[base..base + effective_windows * LIMBS_PER_POINT], effective_windows)
            })
            .collect();

        Ok(results)
    }

    pub fn compute_prepared(
        &mut self,
        prepared: &PreparedHipPippengerMSM,
    ) -> HipResult<Vec<u64>> {
        let clear_threads =
            (prepared.partial_buckets_buf.len_bytes() / std::mem::size_of::<u64>()) as u64;

        let clear_bufs: &[&DeviceBuffer] = &[
            &prepared.partial_buckets_buf,
            &prepared.clear_config_buf,
        ];
        let accum_bufs: &[&DeviceBuffer] = &[
            &prepared.digits_buf,
            &prepared.points_buf,
            &prepared.partial_buckets_buf,
            &prepared.accum_config_buf,
            &prepared.field_buf,
        ];
        let merge_bufs: &[&DeviceBuffer] = &[
            &prepared.partial_buckets_buf,
            &prepared.buckets_buf,
            &prepared.merge_config_buf,
            &prepared.field_buf,
        ];
        let reduc_bufs: &[&DeviceBuffer] = &[
            &prepared.buckets_buf,
            &prepared.window_sums_buf,
            &prepared.reduction_config_buf,
            &prepared.field_buf,
        ];

        self.state.execute_compute_seq(&[
            ("clear_u64_buffer", clear_bufs, clear_threads),
            (
                "bucket_accumulation_by_chunk",
                accum_bufs,
                prepared.accumulation_threads,
            ),
            ("bucket_merge", merge_bufs, prepared.merge_threads),
            (
                "bucket_reduction",
                reduc_bufs,
                prepared.effective_windows as u64,
            ),
        ])?;

        let window_sums: Vec<u64> = self
            .state
            .read_buffer(&prepared.window_sums_buf, prepared.effective_windows * LIMBS_PER_POINT)?;

        Ok(self.combine_windows(&window_sums, prepared.effective_windows))
    }

    // ─── private helpers ──────────────────────────────────────────────────

    fn field_params_buffer(&self) -> HipResult<DeviceBuffer> {
        // Layout: [modulus(4), inv(1), mont_one(4)] = 9 u64s.
        let mut params = [0u64; COORD_LIMBS + 1 + COORD_LIMBS];
        params[..COORD_LIMBS].copy_from_slice(&self.config.field_modulus);
        params[COORD_LIMBS] = self.config.montgomery_inv;
        let mont_one = compute_mont_one(&self.config.field_modulus);
        params[COORD_LIMBS + 1..].copy_from_slice(&mont_one);
        self.state.alloc_buffer_with_data(&params)
    }

    fn combine_windows(&self, window_sums: &[u64], num_windows: usize) -> Vec<u64> {
        if num_windows == 0 {
            return vec![0u64; LIMBS_PER_POINT];
        }
        let field = FieldParams {
            modulus: self.config.field_modulus,
            inv: self.config.montgomery_inv,
        };
        let mut result = JacobianPoint::from_limbs(
            &window_sums[(num_windows - 1) * LIMBS_PER_POINT..num_windows * LIMBS_PER_POINT],
        );
        for window_idx in (0..num_windows - 1).rev() {
            for _ in 0..self.config.window_size {
                result = result.double(&field);
            }
            let base = window_idx * LIMBS_PER_POINT;
            let w = JacobianPoint::from_limbs(&window_sums[base..base + LIMBS_PER_POINT]);
            result = result.add(&w, &field);
        }
        result.to_limbs()
    }
}

// ─── scalar recoding ──────────────────────────────────────────────────────────

fn recode_scalars_signed(
    config: &HipPippengerMSMConfig,
    scalars: &[u64],
    num_scalars: usize,
) -> Vec<i32> {
    let window_size = config.window_size;
    let num_windows = config.num_windows();
    let half_bucket = 1i32 << (window_size - 1);
    let full_bucket = 1i32 << window_size;
    let mask = (1u64 << window_size) - 1;
    let effective_windows = num_windows + 1;
    let mut digits = vec![0i32; num_scalars * effective_windows];

    for scalar_idx in 0..num_scalars {
        let scalar_base = scalar_idx * config.scalar_limbs;
        let mut carry = 0i32;

        for window_idx in 0..num_windows {
            let bit_offset = window_idx * window_size;
            let limb_idx = bit_offset / 64;
            let bit_in_limb = bit_offset % 64;

            let raw_val = if limb_idx < config.scalar_limbs {
                let mut val = (scalars[scalar_base + limb_idx] >> bit_in_limb) & mask;
                if bit_in_limb + window_size > 64 && limb_idx + 1 < config.scalar_limbs {
                    let remaining = bit_in_limb + window_size - 64;
                    val |= (scalars[scalar_base + limb_idx + 1] & ((1u64 << remaining) - 1))
                        << (64 - bit_in_limb);
                }
                val
            } else {
                0
            };

            let window_val = raw_val as i32 + carry;
            let digit = if window_val >= half_bucket {
                carry = 1;
                window_val - full_bucket
            } else {
                carry = 0;
                window_val
            };
            digits[scalar_idx * effective_windows + window_idx] = digit;
        }
        digits[scalar_idx * effective_windows + num_windows] = carry;
    }

    digits
}

// ─── CPU-side field/point arithmetic (for window combination) ─────────────────
// Mirrors the Metal backend's CPU combine_windows exactly.

#[derive(Clone, Copy)]
struct FieldParams {
    modulus: [u64; COORD_LIMBS],
    inv: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct JacobianPoint {
    x: [u64; COORD_LIMBS],
    y: [u64; COORD_LIMBS],
    z: [u64; COORD_LIMBS],
}

impl JacobianPoint {
    fn identity() -> Self {
        Self {
            x: [0; COORD_LIMBS],
            y: [0; COORD_LIMBS],
            z: [0; COORD_LIMBS],
        }
    }

    fn from_limbs(limbs: &[u64]) -> Self {
        let mut x = [0; COORD_LIMBS];
        let mut y = [0; COORD_LIMBS];
        let mut z = [0; COORD_LIMBS];
        x.copy_from_slice(&limbs[..COORD_LIMBS]);
        y.copy_from_slice(&limbs[COORD_LIMBS..2 * COORD_LIMBS]);
        z.copy_from_slice(&limbs[2 * COORD_LIMBS..3 * COORD_LIMBS]);
        Self { x, y, z }
    }

    fn to_limbs(&self) -> Vec<u64> {
        let mut v = Vec::with_capacity(LIMBS_PER_POINT);
        v.extend_from_slice(&self.x);
        v.extend_from_slice(&self.y);
        v.extend_from_slice(&self.z);
        v
    }

    fn is_identity(&self) -> bool {
        self.z == [0; COORD_LIMBS]
    }

    fn double(&self, f: &FieldParams) -> Self {
        if self.is_identity() {
            return self.clone();
        }
        let a = mont_square(&self.x, f);
        let b = mont_square(&self.y, f);
        let c = mont_square(&b, f);
        let d = field_double(
            &field_sub(&field_sub(&mont_square(&field_add(&self.x, &b, f), f), &a, f), &c, f),
            f,
        );
        let e = field_add(&a, &field_double(&a, f), f);
        let ff = mont_square(&e, f);
        let x3 = field_sub(&ff, &field_double(&d, f), f);
        let y3 = field_sub(
            &mont_mul(&e, &field_sub(&d, &x3, f), f),
            &field_double(&field_double(&field_double(&c, f), f), f),
            f,
        );
        let z3 = field_double(&mont_mul(&self.y, &self.z, f), f);
        Self { x: x3, y: y3, z: z3 }
    }

    fn add(&self, other: &Self, f: &FieldParams) -> Self {
        if self.is_identity() {
            return other.clone();
        }
        if other.is_identity() {
            return self.clone();
        }
        let z1z1 = mont_square(&self.z, f);
        let z2z2 = mont_square(&other.z, f);
        let u1 = mont_mul(&self.x, &z2z2, f);
        let u2 = mont_mul(&other.x, &z1z1, f);
        let s1 = mont_mul(&mont_mul(&self.y, &other.z, f), &z2z2, f);
        let s2 = mont_mul(&mont_mul(&other.y, &self.z, f), &z1z1, f);
        let h = field_sub(&u2, &u1, f);
        if h == [0; COORD_LIMBS] {
            if field_sub(&s2, &s1, f) == [0; COORD_LIMBS] {
                return self.double(f);
            }
            return Self::identity();
        }
        let i = mont_square(&field_double(&h, f), f);
        let j = mont_mul(&h, &i, f);
        let r = field_double(&field_sub(&s2, &s1, f), f);
        let v = mont_mul(&u1, &i, f);
        let x3 = field_sub(&field_sub(&mont_square(&r, f), &j, f), &field_double(&v, f), f);
        let y3 = field_sub(
            &mont_mul(&r, &field_sub(&v, &x3, f), f),
            &field_double(&mont_mul(&s1, &j, f), f),
            f,
        );
        let z3 = mont_mul(
            &field_sub(
                &field_sub(&mont_square(&field_add(&self.z, &other.z, f), f), &z1z1, f),
                &z2z2,
                f,
            ),
            &h,
            f,
        );
        Self { x: x3, y: y3, z: z3 }
    }
}

fn bigint_add(a: &[u64; COORD_LIMBS], b: &[u64; COORD_LIMBS]) -> ([u64; COORD_LIMBS], u64) {
    let mut result = [0u64; COORD_LIMBS];
    let mut carry = 0u64;
    for i in 0..COORD_LIMBS {
        let (s1, c1) = a[i].overflowing_add(b[i]);
        let (s2, c2) = s1.overflowing_add(carry);
        result[i] = s2;
        carry = u64::from(c1) + u64::from(c2);
    }
    (result, carry)
}

fn bigint_sub(a: &[u64; COORD_LIMBS], b: &[u64; COORD_LIMBS]) -> ([u64; COORD_LIMBS], u64) {
    let mut result = [0u64; COORD_LIMBS];
    let mut borrow = 0u64;
    for i in 0..COORD_LIMBS {
        let (d1, b1) = a[i].overflowing_sub(b[i]);
        let (d2, b2) = d1.overflowing_sub(borrow);
        result[i] = d2;
        borrow = u64::from(b1) + u64::from(b2);
    }
    (result, borrow)
}

fn field_add(a: &[u64; COORD_LIMBS], b: &[u64; COORD_LIMBS], f: &FieldParams) -> [u64; COORD_LIMBS] {
    let (sum, carry) = bigint_add(a, b);
    let (reduced, borrow) = bigint_sub(&sum, &f.modulus);
    if carry != 0 || borrow == 0 { reduced } else { sum }
}

fn field_sub(a: &[u64; COORD_LIMBS], b: &[u64; COORD_LIMBS], f: &FieldParams) -> [u64; COORD_LIMBS] {
    let (diff, borrow) = bigint_sub(a, b);
    if borrow != 0 { bigint_add(&diff, &f.modulus).0 } else { diff }
}

fn field_double(a: &[u64; COORD_LIMBS], f: &FieldParams) -> [u64; COORD_LIMBS] {
    field_add(a, a, f)
}

fn mont_reduce(t: &[u64; COORD_LIMBS * 2], f: &FieldParams) -> [u64; COORD_LIMBS] {
    let mut tmp = *t;
    for i in 0..COORD_LIMBS {
        let m = tmp[i].wrapping_mul(f.inv);
        let mut carry = 0u64;
        for j in 0..COORD_LIMBS {
            let wide = (m as u128) * (f.modulus[j] as u128);
            let lo = wide as u64;
            let hi = (wide >> 64) as u64;
            let (s1, c1) = tmp[i + j].overflowing_add(lo);
            let (s2, c2) = s1.overflowing_add(carry);
            tmp[i + j] = s2;
            carry = hi + u64::from(c1) + u64::from(c2);
        }
        for j in COORD_LIMBS..(COORD_LIMBS * 2 - i) {
            let (s, c) = tmp[i + j].overflowing_add(carry);
            tmp[i + j] = s;
            carry = u64::from(c);
            if carry == 0 {
                break;
            }
        }
    }
    let mut result = [0u64; COORD_LIMBS];
    result.copy_from_slice(&tmp[COORD_LIMBS..]);
    let (reduced, borrow) = bigint_sub(&result, &f.modulus);
    if borrow == 0 { reduced } else { result }
}

fn mont_mul(a: &[u64; COORD_LIMBS], b: &[u64; COORD_LIMBS], f: &FieldParams) -> [u64; COORD_LIMBS] {
    let mut product = [0u64; COORD_LIMBS * 2];
    for i in 0..COORD_LIMBS {
        let mut carry = 0u64;
        for j in 0..COORD_LIMBS {
            let wide = (a[i] as u128) * (b[j] as u128);
            let lo = wide as u64;
            let hi = (wide >> 64) as u64;
            let (s1, c1) = product[i + j].overflowing_add(lo);
            let (s2, c2) = s1.overflowing_add(carry);
            product[i + j] = s2;
            carry = hi + u64::from(c1) + u64::from(c2);
        }
        let (s, _) = product[i + COORD_LIMBS].overflowing_add(carry);
        product[i + COORD_LIMBS] = s;
    }
    mont_reduce(&product, f)
}

fn mont_square(a: &[u64; COORD_LIMBS], f: &FieldParams) -> [u64; COORD_LIMBS] {
    mont_mul(a, a, f)
}

/// Compute R = 2^256 mod modulus (the Montgomery representation of 1).
///
/// Uses 256 modular doublings starting from 1.  Called once at init time —
/// cost is negligible.  Limbs are in little-endian order (limb[0] = LSW).
fn compute_mont_one(modulus: &[u64; COORD_LIMBS]) -> [u64; COORD_LIMBS] {
    let mut r = [1u64, 0, 0, 0]; // 1 in LE
    for _ in 0..256 {
        let mut carry = 0u64;
        let mut doubled = [0u64; COORD_LIMBS];
        for i in 0..COORD_LIMBS {
            let (a, c1) = r[i].overflowing_add(r[i]);
            let (b, c2) = a.overflowing_add(carry);
            doubled[i] = b;
            carry = u64::from(c1) + u64::from(c2);
        }
        let mut borrow = 0u64;
        let mut sub = [0u64; COORD_LIMBS];
        for i in 0..COORD_LIMBS {
            let (d, b1) = doubled[i].overflowing_sub(modulus[i]);
            let (e, b2) = d.overflowing_sub(borrow);
            sub[i] = e;
            borrow = u64::from(b1) + u64::from(b2);
        }
        r = if carry != 0 || borrow == 0 { sub } else { doubled };
    }
    r
}

fn montgomery_inv64(m0: u64) -> u64 {
    let mut inv = 1u64;
    for _ in 0..6 {
        inv = inv.wrapping_mul(2u64.wrapping_sub(m0.wrapping_mul(inv)));
    }
    inv.wrapping_neg()
}

// ─── unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pallas_and_vesta_inverses_match_modulus_lsb() {
        for config in [HipPippengerMSMConfig::pallas(), HipPippengerMSMConfig::vesta()] {
            assert_eq!(
                config.field_modulus[0].wrapping_mul(config.montgomery_inv),
                u64::MAX,
                "Montgomery inverse must satisfy m * m⁻¹ ≡ -1 (mod 2^64)"
            );
        }
    }

    #[test]
    fn config_window_counts_are_256_bit() {
        let config = HipPippengerMSMConfig {
            window_size: 8,
            ..HipPippengerMSMConfig::pallas()
        };
        assert_eq!(config.num_windows(), 32);
        assert_eq!(config.num_buckets(), 128);
    }

    #[test]
    fn recoding_uses_extra_carry_window() {
        let config = HipPippengerMSMConfig {
            window_size: 4,
            ..HipPippengerMSMConfig::pallas()
        };
        let scalars = [u64::MAX; COORD_LIMBS];
        let digits = recode_scalars_signed(&config, &scalars, 1);
        assert_eq!(digits.len(), config.num_windows() + 1);
        assert_eq!(*digits.last().unwrap(), 1);
    }

    #[test]
    fn identity_arithmetic_is_stable() {
        let field = FieldParams {
            modulus: PALLAS_MODULUS,
            inv: montgomery_inv64(PALLAS_MODULUS[0]),
        };
        let id = JacobianPoint::identity();
        assert_eq!(id.double(&field), id);
        let p = JacobianPoint { x: [1, 2, 3, 4], y: [5, 6, 7, 8], z: [1, 0, 0, 0] };
        assert_eq!(id.add(&p, &field), p);
    }
}
