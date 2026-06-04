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

/// Must match `MERGE_BLOCK_SIZE` in the HIP shader.
const MERGE_BLOCK_SIZE: u32 = 64;
/// LDS shared memory per `bucket_merge` block: 64 threads × 12 limbs × 8 bytes.
const MERGE_SHM_BYTES: u32 = MERGE_BLOCK_SIZE * LIMBS_PER_POINT as u32 * 8;

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
            // window_size = 4: num_buckets = 8 → 8 × 12 u64 = 192 VGPRs < 256/lane limit →
            // zero register spilling on RDNA3.  `compute()` always calls
            // `config_for_num_points()` to get the best chunk_size for the actual input.
            window_size: 4,
            scalar_limbs: COORD_LIMBS,
            bits_per_limb: 64,
            montgomery_inv: montgomery_inv64(field_modulus[0]),
            field_modulus,
            // Conservative fallback; `compute()` / `config_for_num_points()` override this.
            chunk_size: 64,
        }
    }

    pub fn optimal_window_size(_num_points: usize) -> usize {
        // Capped at 4: num_buckets = 2^(4-1) = 8 → 8 × 12 u64 = 192 VGPRs per thread.
        // RDNA3 allows 256 VGPRs/lane; staying under this limit eliminates register
        // spilling to scratch memory, which was the dominant bottleneck at larger windows.
        // A fixed window=4 simplifies tuning; all scalar sizes use the same path.
        4
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
        // Target ~8 wavefronts per CU for good latency hiding with window=4.
        // With ~192 VGPRs per thread, GFX11 can sustain 4-8 wavefronts/CU.
        // Using 8× means chunk_size shrinks → more (window,chunk) pairs → better
        // GPU saturation compared to the old 4× factor with 60 hardcoded CUs.
        let target_threads = cu_count * 64 * 8;
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
    accum_config_buf: DeviceBuffer,
    merge_config_buf: DeviceBuffer,
    /// Config for the fused bucket_reduce_combine kernel: [num_windows, num_buckets, window_size].
    rc_config_buf: DeviceBuffer,
    field_buf: DeviceBuffer,
    result_buf: DeviceBuffer,
    effective_windows: usize,
    num_chunks: usize,
    num_buckets: usize,
    window_size: usize,
}

// ─── persistent bases ─────────────────────────────────────────────────────────

/// Affine input points pre-uploaded to GPU memory for reuse across many MSMs.
///
/// Create once via [`HipPippengerMSM::prepare_bases`]; reuse with
/// [`HipPippengerMSM::compute_with_bases`].  Avoids the PCIe transfer cost of
/// re-uploading the same base points for every scalar change.
pub struct HipPippengerBases {
    points_buf: DeviceBuffer,
    field_buf: DeviceBuffer,
    /// Number of affine input points (= `x/y` pairs).
    pub num_points: usize,
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
        let cu_count = HipState::cu_count();
        let chunk_size =
            HipPippengerMSMConfig::optimal_chunk_size(num_points, effective_windows, cu_count);
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
        // Select the curve-specific compile option so hipRTC can inline the
        // modulus constants and generate the optimised mont_mul.
        let compile_opt = if self.config.field_modulus == PALLAS_MODULUS {
            Some("-DPALLAS_CURVE")
        } else if self.config.field_modulus == VESTA_MODULUS {
            Some("-DVESTA_CURVE")
        } else {
            None
        };
        let opts_vec: Vec<&str> = compile_opt.into_iter().collect();
        self.state
            .load_source_with_opts(PIPPENGER_MSM_HIP_SOURCE, "pippenger_msm.hip", &opts_vec)?;
        self.state.prepare_function("bucket_accumulation_by_chunk")?;
        self.state.prepare_function("bucket_merge")?;
        self.state.prepare_function("bucket_reduce_combine")?;
        // Keep the standalone kernels for compatibility / manual use.
        self.state.prepare_function("bucket_reduction")?;
        self.state.prepare_function("combine_windows_kernel")?;
        self.initialized = true;
        Ok(())
    }

    /// Compute `Σ scalars[i] * points[i]`.
    ///
    /// `scalars`: flat `[s0_l0, ..., s0_l3, s1_l0, ...]`.
    /// `points`:  flat `[x0_l0..x0_l3, y0_l0..y0_l3, x1_l0..]` (affine, 8 limbs/point).
    ///
    /// Automatically derives the optimal `window_size` and `chunk_size` for the
    /// given input length.  `self.config` is restored after the call.
    pub fn compute(&mut self, scalars: &[u64], points: &[u64]) -> HipResult<Vec<u64>> {
        let num_points = points.len() / LIMBS_PER_AFFINE;
        let optimal = self.config_for_num_points(num_points);
        let prev_config = std::mem::replace(&mut self.config, optimal);
        let result = self.prepare(scalars, points).and_then(|p| self.compute_prepared(&p));
        self.config = prev_config;
        result
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
        // Fused reduce+combine config: [num_windows, num_buckets, window_size]
        let rc_config_buf = self.state.alloc_buffer_with_data(&[
            effective_windows as u32,
            num_buckets as u32,
            self.config.window_size as u32,
        ])?;
        let field_buf = self.field_params_buffer()?;

        let result_buf =
            self.state.alloc_buffer(LIMBS_PER_POINT * std::mem::size_of::<u64>())?;

        Ok(PreparedHipPippengerMSM {
            digits_buf,
            points_buf,
            partial_buckets_buf,
            buckets_buf,
            accum_config_buf,
            merge_config_buf,
            rc_config_buf,
            field_buf,
            result_buf,
            effective_windows,
            num_chunks,
            num_buckets,
            window_size: self.config.window_size,
        })
    }

    /// Compute multiple MSMs in a single batched GPU kernel launch.
    ///
    /// All entries in `batch` must have the same number of points and scalars.
    /// Results are returned in the same order as the input.
    ///
    /// Validates every entry strictly and auto-derives optimal `window_size` /
    /// `chunk_size` for the given input size.  `self.config` is not modified.
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

        // ── strict per-entry validation ─────────────────────────────────────
        let (s0, p0) = batch[0];
        if s0.len() % self.config.scalar_limbs != 0 {
            return Err(HipError::InvalidInputSize {
                expected: self.config.scalar_limbs,
                actual: s0.len(),
            });
        }
        if p0.len() % LIMBS_PER_AFFINE != 0 {
            return Err(HipError::InvalidInputSize {
                expected: LIMBS_PER_AFFINE,
                actual: p0.len(),
            });
        }
        let num_scalars = s0.len() / self.config.scalar_limbs;
        let num_points  = p0.len() / LIMBS_PER_AFFINE;
        if num_scalars == 0 {
            return Err(HipError::EmptyInput);
        }
        if num_scalars != num_points {
            return Err(HipError::LengthMismatch(num_scalars, num_points));
        }
        for (s, p) in batch.iter().skip(1) {
            if s.len() % self.config.scalar_limbs != 0 {
                return Err(HipError::InvalidInputSize {
                    expected: self.config.scalar_limbs,
                    actual: s.len(),
                });
            }
            if p.len() % LIMBS_PER_AFFINE != 0 {
                return Err(HipError::InvalidInputSize {
                    expected: LIMBS_PER_AFFINE,
                    actual: p.len(),
                });
            }
            let ns = s.len() / self.config.scalar_limbs;
            let np = p.len() / LIMBS_PER_AFFINE;
            if ns != num_scalars {
                return Err(HipError::LengthMismatch(num_scalars, ns));
            }
            if np != num_scalars {
                return Err(HipError::LengthMismatch(num_scalars, np));
            }
        }

        // ── auto-derive optimal config (does not mutate self.config) ────────
        let config = self.config_for_num_points(num_scalars);
        let num_buckets       = config.num_buckets();
        let effective_windows = config.num_windows() + 1;
        let num_chunks        = num_scalars.div_ceil(config.chunk_size);

        // ── recode scalars directly — no intermediate all_scalars_raw copy ──
        let mut all_digits: Vec<i8> =
            Vec::with_capacity(num_msms * effective_windows * num_scalars);
        for (s, _) in batch {
            let digits = recode_scalars_signed(&config, s, num_scalars);
            all_digits.extend_from_slice(&digits);
        }

        // ── concatenate points for a single GPU upload ──────────────────────
        let mut all_points: Vec<u64> =
            Vec::with_capacity(num_msms * num_scalars * LIMBS_PER_AFFINE);
        for (_, p) in batch {
            all_points.extend_from_slice(p);
        }

        let digits_buf = self.state.alloc_buffer_with_data(&all_digits)?;
        let points_buf = self.state.alloc_buffer_with_data(&all_points)?;

        // ── intermediate GPU buffers ─────────────────────────────────────────
        let partial_len =
            num_msms * effective_windows * num_chunks * num_buckets * LIMBS_PER_POINT;
        let partial_buckets_buf =
            self.state.alloc_buffer(partial_len * std::mem::size_of::<u64>())?;

        let buckets_len = num_msms * effective_windows * num_buckets * LIMBS_PER_POINT;
        let buckets_buf =
            self.state.alloc_buffer(buckets_len * std::mem::size_of::<u64>())?;

        let result_buf = self.state.alloc_buffer(
            num_msms * LIMBS_PER_POINT * std::mem::size_of::<u64>(),
        )?;

        // ── config buffers ───────────────────────────────────────────────────
        let accum_config_buf = self.state.alloc_buffer_with_data(&[
            num_scalars as u32,
            effective_windows as u32,
            num_buckets as u32,
            config.chunk_size as u32,
            num_chunks as u32,
        ])?;
        let merge_config_buf = self.state.alloc_buffer_with_data(&[
            effective_windows as u32,
            num_buckets as u32,
            num_chunks as u32,
        ])?;
        let rc_config_buf = self.state.alloc_buffer_with_data(&[
            effective_windows as u32,
            num_buckets as u32,
            config.window_size as u32,
        ])?;
        let field_buf = self.field_params_buffer()?;

        // ── kernel argument slices ───────────────────────────────────────────
        let accum_bufs: &[&DeviceBuffer] = &[
            &digits_buf, &points_buf, &partial_buckets_buf, &accum_config_buf, &field_buf,
        ];
        let merge_bufs: &[&DeviceBuffer] = &[
            &partial_buckets_buf, &buckets_buf, &merge_config_buf, &field_buf,
        ];
        let rc_bufs: &[&DeviceBuffer] = &[
            &buckets_buf, &result_buf, &rc_config_buf, &field_buf,
        ];

        // ── kernel launches ──────────────────────────────────────────────────
        let accum_gx = ((effective_windows * num_chunks) as u32 + 255) / 256;
        let merge_gx = (effective_windows * num_buckets) as u32;
        // Fused reduce+combine: one block per MSM, block has effective_windows threads.
        let rc_block = effective_windows as u32;
        let rc_shm   = rc_block * LIMBS_PER_POINT as u32 * 8;

        // First two kernels: grid_y = num_msms (each MSM gets its own y-slice).
        self.state.execute_kernels_2d(
            &[
                ("bucket_accumulation_by_chunk", accum_bufs, accum_gx, 256,              0),
                ("bucket_merge",                 merge_bufs, merge_gx, MERGE_BLOCK_SIZE, MERGE_SHM_BYTES),
            ],
            num_msms as u32,
        )?;

        // Fused reduce+combine: grid_x=1, grid_y=num_msms → one block per MSM.
        self.state.execute_kernels_2d(
            &[("bucket_reduce_combine", rc_bufs, 1, rc_block, rc_shm)],
            num_msms as u32,
        )?;

        // ── read back one Jacobian point per MSM ────────────────────────────
        let all_results: Vec<u64> =
            self.state.read_buffer(&result_buf, num_msms * LIMBS_PER_POINT)?;

        Ok((0..num_msms)
            .map(|i| all_results[i * LIMBS_PER_POINT..(i + 1) * LIMBS_PER_POINT].to_vec())
            .collect())
    }

    pub fn compute_prepared(
        &mut self,
        prepared: &PreparedHipPippengerMSM,
    ) -> HipResult<Vec<u64>> {
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
        let rc_bufs: &[&DeviceBuffer] = &[
            &prepared.buckets_buf,
            &prepared.result_buf,
            &prepared.rc_config_buf,
            &prepared.field_buf,
        ];

        let accum_gx  = ((prepared.effective_windows * prepared.num_chunks) as u32 + 255) / 256;
        let merge_gx  = (prepared.effective_windows * prepared.num_buckets) as u32;
        // Fused reduce+combine: one block of effective_windows threads per MSM.
        let rc_block  = prepared.effective_windows as u32;
        let rc_shm    = rc_block * LIMBS_PER_POINT as u32 * 8; // dynamic shared memory bytes

        self.state.execute_kernels_2d(
            &[
                ("bucket_accumulation_by_chunk", accum_bufs, accum_gx, 256,              0),
                ("bucket_merge",                 merge_bufs, merge_gx, MERGE_BLOCK_SIZE, MERGE_SHM_BYTES),
                ("bucket_reduce_combine",        rc_bufs,    1,        rc_block,         rc_shm),
            ],
            1,
        )?;

        self.state.read_buffer(&prepared.result_buf, LIMBS_PER_POINT)
    }

    // ─── persistent-bases API ────────────────────────────────────────────

    /// Upload affine input points to GPU memory for repeated reuse.
    ///
    /// `points` is flat `[x0_l0..l3, y0_l0..l3, x1_l0..l3, …]` in Montgomery form.
    /// Call this once per unique point set; then call [`compute_with_bases`] for
    /// each new set of scalars without paying the PCIe transfer cost again.
    pub fn prepare_bases(&mut self, points: &[u64]) -> HipResult<HipPippengerBases> {
        if !self.initialized {
            self.initialize()?;
        }
        if points.is_empty() {
            return Err(HipError::EmptyInput);
        }
        if points.len() % LIMBS_PER_AFFINE != 0 {
            return Err(HipError::InvalidInputSize {
                expected: LIMBS_PER_AFFINE,
                actual: points.len(),
            });
        }
        let num_points = points.len() / LIMBS_PER_AFFINE;
        let points_buf = self.state.alloc_buffer_with_data(points)?;
        let field_buf  = self.field_params_buffer()?;
        Ok(HipPippengerBases { points_buf, field_buf, num_points })
    }

    /// Compute `Σ scalars[i] * bases[i]` using pre-uploaded bases.
    ///
    /// Derives the optimal `window_size` and `chunk_size` from `bases.num_points`
    /// without mutating `self.config`.  Only scalars are transferred to the GPU.
    pub fn compute_with_bases(
        &mut self,
        scalars: &[u64],
        bases: &HipPippengerBases,
    ) -> HipResult<Vec<u64>> {
        if !self.initialized {
            self.initialize()?;
        }
        if scalars.len() % self.config.scalar_limbs != 0 {
            return Err(HipError::InvalidInputSize {
                expected: self.config.scalar_limbs,
                actual: scalars.len(),
            });
        }
        let num_scalars = scalars.len() / self.config.scalar_limbs;
        if num_scalars == 0 {
            return Err(HipError::EmptyInput);
        }
        if num_scalars != bases.num_points {
            return Err(HipError::LengthMismatch(num_scalars, bases.num_points));
        }

        // Derive optimal config without mutating self.config.
        let config           = self.config_for_num_points(bases.num_points);
        let num_windows      = config.num_windows();
        let effective_windows = num_windows + 1;
        let num_buckets      = config.num_buckets();
        let num_chunks       = num_scalars.div_ceil(config.chunk_size);

        // Only scalars need to be encoded and uploaded.
        let signed_digits = recode_scalars_signed(&config, scalars, num_scalars);
        let digits_buf    = self.state.alloc_buffer_with_data(&signed_digits)?;

        // Workspace buffers.
        let partial_len = effective_windows * num_chunks * num_buckets * LIMBS_PER_POINT;
        let partial_buckets_buf =
            self.state.alloc_buffer(partial_len * std::mem::size_of::<u64>())?;

        let buckets_len = effective_windows * num_buckets * LIMBS_PER_POINT;
        let buckets_buf =
            self.state.alloc_buffer(buckets_len * std::mem::size_of::<u64>())?;

        let result_buf = self.state.alloc_buffer(LIMBS_PER_POINT * std::mem::size_of::<u64>())?;

        // Config buffers.
        let accum_config_buf = self.state.alloc_buffer_with_data(&[
            num_scalars as u32,
            effective_windows as u32,
            num_buckets as u32,
            config.chunk_size as u32,
            num_chunks as u32,
        ])?;
        let merge_config_buf = self.state.alloc_buffer_with_data(&[
            effective_windows as u32,
            num_buckets as u32,
            num_chunks as u32,
        ])?;
        // Fused reduce+combine config: [num_windows, num_buckets, window_size]
        let rc_config_buf = self.state.alloc_buffer_with_data(&[
            effective_windows as u32,
            num_buckets as u32,
            config.window_size as u32,
        ])?;

        let accum_bufs: &[&DeviceBuffer] = &[
            &digits_buf, &bases.points_buf, &partial_buckets_buf,
            &accum_config_buf, &bases.field_buf,
        ];
        let merge_bufs: &[&DeviceBuffer] = &[
            &partial_buckets_buf, &buckets_buf, &merge_config_buf, &bases.field_buf,
        ];
        let rc_bufs: &[&DeviceBuffer] = &[
            &buckets_buf, &result_buf, &rc_config_buf, &bases.field_buf,
        ];

        let accum_gx = ((effective_windows * num_chunks) as u32 + 255) / 256;
        let merge_gx = (effective_windows * num_buckets) as u32;
        // Fused reduce+combine: one block of effective_windows threads.
        let rc_block = effective_windows as u32;
        let rc_shm   = rc_block * LIMBS_PER_POINT as u32 * 8;

        self.state.execute_kernels_2d(
            &[
                ("bucket_accumulation_by_chunk", accum_bufs, accum_gx, 256,              0),
                ("bucket_merge",                 merge_bufs, merge_gx, MERGE_BLOCK_SIZE, MERGE_SHM_BYTES),
                ("bucket_reduce_combine",        rc_bufs,    1,        rc_block,         rc_shm),
            ],
            1,
        )?;

        self.state.read_buffer(&result_buf, LIMBS_PER_POINT)
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

}

// ─── scalar recoding ──────────────────────────────────────────────────────────

/// Recode 256-bit scalars into signed window digits using i8 storage.
///
/// Produces transposed layout: `digits[window_idx * num_scalars + scalar_idx]`.
/// This lets the GPU load a tile of consecutive scalars for a fixed window in a
/// single coalesced transaction.
///
/// For `window_size ≤ 7` the digit range is −64 .. +64, which fits in i8.
/// `window_size = 4` (the default) produces −8 .. +8.
fn recode_scalars_signed(
    config: &HipPippengerMSMConfig,
    scalars: &[u64],
    num_scalars: usize,
) -> Vec<i8> {
    let window_size = config.window_size;
    debug_assert!(
        window_size <= 7,
        "i8 digit encoding requires window_size ≤ 7 (got {})",
        window_size
    );
    let num_windows   = config.num_windows();
    let half_bucket   = 1i32 << (window_size - 1);
    let full_bucket   = 1i32 << window_size;
    let mask          = (1u64 << window_size) - 1;
    let eff_windows   = num_windows + 1;

    let mut digits = vec![0i8; num_scalars * eff_windows];

    for scalar_idx in 0..num_scalars {
        let scalar_base = scalar_idx * config.scalar_limbs;
        let mut carry   = 0i32;

        for window_idx in 0..num_windows {
            let bit_offset  = window_idx * window_size;
            let limb_idx    = bit_offset / 64;
            let bit_in_limb = bit_offset % 64;

            let raw_val = if limb_idx < config.scalar_limbs {
                let mut val = (scalars[scalar_base + limb_idx] >> bit_in_limb) & mask;
                if bit_in_limb + window_size > 64 && limb_idx + 1 < config.scalar_limbs {
                    let remaining = bit_in_limb + window_size - 64;
                    val |= (scalars[scalar_base + limb_idx + 1]
                        & ((1u64 << remaining) - 1))
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

            // Transposed layout: [window][scalar]
            digits[window_idx * num_scalars + scalar_idx] = digit as i8;
        }
        digits[num_windows * num_scalars + scalar_idx] = carry as i8;
    }

    digits
}

// ─── CPU-side field/point arithmetic ─────────────────────────────────────────
// Used only by the unit tests below (the `combine_windows` step now runs on GPU).

#[allow(dead_code)]
#[derive(Clone, Copy)]
struct FieldParams {
    modulus: [u64; COORD_LIMBS],
    inv: u64,
}

#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq, Eq)]
struct JacobianPoint {
    x: [u64; COORD_LIMBS],
    y: [u64; COORD_LIMBS],
    z: [u64; COORD_LIMBS],
}

#[allow(dead_code)]
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

#[allow(dead_code)]
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

#[allow(dead_code)]
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

#[allow(dead_code)]
fn field_add(a: &[u64; COORD_LIMBS], b: &[u64; COORD_LIMBS], f: &FieldParams) -> [u64; COORD_LIMBS] {
    let (sum, carry) = bigint_add(a, b);
    let (reduced, borrow) = bigint_sub(&sum, &f.modulus);
    if carry != 0 || borrow == 0 { reduced } else { sum }
}

#[allow(dead_code)]
fn field_sub(a: &[u64; COORD_LIMBS], b: &[u64; COORD_LIMBS], f: &FieldParams) -> [u64; COORD_LIMBS] {
    let (diff, borrow) = bigint_sub(a, b);
    if borrow != 0 { bigint_add(&diff, &f.modulus).0 } else { diff }
}

#[allow(dead_code)]
fn field_double(a: &[u64; COORD_LIMBS], f: &FieldParams) -> [u64; COORD_LIMBS] {
    field_add(a, a, f)
}

#[allow(dead_code)]
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

#[allow(dead_code)]
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

#[allow(dead_code)]
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
        // effective_windows = num_windows + 1 (carry window)
        assert_eq!(digits.len(), config.num_windows() + 1);
        // carry propagates to the last (extra) window
        assert_eq!(*digits.last().unwrap(), 1i8);
    }

    #[test]
    fn batch_inverse_matches_cpu_pallas() {
        let mut msm = HipPippengerMSM::new_pallas().unwrap();
        msm.initialize().unwrap();
        msm.state.prepare_function("batch_inverse_test").unwrap();

        let f = FieldParams {
            modulus: PALLAS_MODULUS,
            inv: montgomery_inv64(PALLAS_MODULUS[0]),
        };
        let mont_one = compute_mont_one(&PALLAS_MODULUS);

        // n deterministic pseudo-random nonzero residues in [1, p).
        let n = 100usize;
        let mut x = 0x0123_4567_89ab_cdefu64;
        let mut elems: Vec<[u64; COORD_LIMBS]> = Vec::with_capacity(n);
        let mut vals: Vec<u64> = Vec::with_capacity(n * COORD_LIMBS);
        for _ in 0..n {
            let mut e = [0u64; COORD_LIMBS];
            for limb in e.iter_mut() {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
                *limb = x;
            }
            // e[3] < 2^62 ⇒ e < 2^254 < p, guaranteeing a valid residue.
            e[3] &= 0x3fff_ffff_ffff_ffff;
            if e == [0u64; COORD_LIMBS] {
                e[0] = 1;
            }
            elems.push(e);
            vals.extend_from_slice(&e);
        }

        let vals_buf = msm.state.alloc_buffer_with_data(&vals).unwrap();
        let out_buf = msm
            .state
            .alloc_buffer(n * COORD_LIMBS * std::mem::size_of::<u64>())
            .unwrap();
        let cfg_buf = msm.state.alloc_buffer_with_data(&[n as u32]).unwrap();
        let field_buf = msm.field_params_buffer().unwrap();

        msm.state
            .execute_compute(
                "batch_inverse_test",
                &[&vals_buf, &out_buf, &cfg_buf, &field_buf],
                1,
            )
            .unwrap();
        let out: Vec<u64> = msm.state.read_buffer(&out_buf, n * COORD_LIMBS).unwrap();

        for (i, a) in elems.iter().enumerate() {
            let mut b = [0u64; COORD_LIMBS];
            b.copy_from_slice(&out[i * COORD_LIMBS..(i + 1) * COORD_LIMBS]);
            // batch_inverse returns b with mont_mul(a, b) == mont_one (= R mod p).
            assert_eq!(mont_mul(a, &b, &f), mont_one, "wrong inverse at element {i}");
        }
    }

    fn mont_inverse_cpu(a: &[u64; COORD_LIMBS], f: &FieldParams) -> [u64; COORD_LIMBS] {
        let (exp, _) = bigint_sub(&f.modulus, &[2, 0, 0, 0]);
        let mut result = compute_mont_one(&f.modulus);
        let mut base = *a;
        for limb in exp.iter() {
            let mut e = *limb;
            for _ in 0..64 {
                if e & 1 == 1 {
                    result = mont_mul(&result, &base, f);
                }
                base = mont_square(&base, f);
                e >>= 1;
            }
        }
        result
    }

    #[test]
    fn batch_affine_add_matches_cpu_pallas() {
        let mut msm = HipPippengerMSM::new_pallas().unwrap();
        msm.initialize().unwrap();
        msm.state.prepare_function("batch_affine_add_test").unwrap();

        let f = FieldParams {
            modulus: PALLAS_MODULUS,
            inv: montgomery_inv64(PALLAS_MODULUS[0]),
        };

        let n = 80usize;
        let mut x = 0x0bad_c0de_dead_beefu64;
        let mut next = || -> [u64; COORD_LIMBS] {
            let mut e = [0u64; COORD_LIMBS];
            for limb in e.iter_mut() {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
                *limb = x;
            }
            e[3] &= 0x3fff_ffff_ffff_ffff; // < 2^254 < p
            e
        };

        let mut pin: Vec<u64> = Vec::new();
        let mut qin: Vec<u64> = Vec::new();
        let mut p_pts: Vec<([u64; 4], [u64; 4])> = Vec::new();
        let mut q_pts: Vec<([u64; 4], [u64; 4])> = Vec::new();
        for _ in 0..n {
            let px = next();
            let py = next();
            let mut qx = next();
            if qx == px {
                qx[0] ^= 1;
            }
            let qy = next();
            pin.extend_from_slice(&px);
            pin.extend_from_slice(&py);
            qin.extend_from_slice(&qx);
            qin.extend_from_slice(&qy);
            p_pts.push((px, py));
            q_pts.push((qx, qy));
        }

        let pin_buf = msm.state.alloc_buffer_with_data(&pin).unwrap();
        let qin_buf = msm.state.alloc_buffer_with_data(&qin).unwrap();
        let rout_buf = msm
            .state
            .alloc_buffer(n * LIMBS_PER_AFFINE * std::mem::size_of::<u64>())
            .unwrap();
        let cfg_buf = msm.state.alloc_buffer_with_data(&[n as u32]).unwrap();
        let field_buf = msm.field_params_buffer().unwrap();

        msm.state
            .execute_compute(
                "batch_affine_add_test",
                &[&pin_buf, &qin_buf, &rout_buf, &cfg_buf, &field_buf],
                1,
            )
            .unwrap();
        let rout: Vec<u64> = msm
            .state
            .read_buffer(&rout_buf, n * LIMBS_PER_AFFINE)
            .unwrap();

        for i in 0..n {
            let (px, py) = p_pts[i];
            let (qx, qy) = q_pts[i];
            // CPU affine add (same formula as the GPU kernel).
            let denom = field_sub(&qx, &px, &f);
            let dinv = mont_inverse_cpu(&denom, &f);
            let num = field_sub(&qy, &py, &f);
            let lambda = mont_mul(&num, &dinv, &f);
            let l2 = mont_square(&lambda, &f);
            let xr = field_sub(&field_sub(&l2, &px, &f), &qx, &f);
            let yr = field_sub(&mont_mul(&lambda, &field_sub(&px, &xr, &f), &f), &py, &f);

            let base = i * LIMBS_PER_AFFINE;
            assert_eq!(&rout[base..base + 4], &xr, "x mismatch at pair {i}");
            assert_eq!(&rout[base + 4..base + 8], &yr, "y mismatch at pair {i}");
        }
    }

    #[test]
    fn bucket_counting_sort_matches_cpu() {
        let mut msm = HipPippengerMSM::new_pallas().unwrap();
        msm.initialize().unwrap();
        msm.state.prepare_function("bucket_counting_sort").unwrap();

        let config = HipPippengerMSMConfig {
            window_size: 4,
            ..HipPippengerMSMConfig::pallas()
        };
        let ns = 50usize;
        let nb = config.num_buckets();
        let nw = config.num_windows() + 1; // include carry window

        // Deterministic pseudo-random scalars.
        let mut x = 0xfeed_face_cafe_d00du64;
        let mut scalars: Vec<u64> = Vec::with_capacity(ns * COORD_LIMBS);
        for _ in 0..ns * COORD_LIMBS {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
            scalars.push(x);
        }
        let digits = recode_scalars_signed(&config, &scalars, ns); // [window][scalar]

        let digits_buf = msm.state.alloc_buffer_with_data(&digits).unwrap();
        let idx_buf = msm
            .state
            .alloc_buffer(nw * ns * std::mem::size_of::<u32>())
            .unwrap();
        let sign_buf = msm.state.alloc_buffer(nw * ns).unwrap();
        let start_buf = msm
            .state
            .alloc_buffer(nw * nb * std::mem::size_of::<u32>())
            .unwrap();
        let counts_buf = msm
            .state
            .alloc_buffer(nw * nb * std::mem::size_of::<u32>())
            .unwrap();
        let cfg_buf = msm
            .state
            .alloc_buffer_with_data(&[ns as u32, nw as u32, nb as u32])
            .unwrap();

        msm.state
            .execute_compute(
                "bucket_counting_sort",
                &[&digits_buf, &idx_buf, &sign_buf, &start_buf, &counts_buf, &cfg_buf],
                nw as u64,
            )
            .unwrap();

        let g_idx: Vec<u32> = msm.state.read_buffer(&idx_buf, nw * ns).unwrap();
        let g_sign: Vec<i8> = msm.state.read_buffer(&sign_buf, nw * ns).unwrap();
        let g_start: Vec<u32> = msm.state.read_buffer(&start_buf, nw * nb).unwrap();
        let g_counts: Vec<u32> = msm.state.read_buffer(&counts_buf, nw * nb).unwrap();

        for w in 0..nw {
            let mut counts = vec![0u32; nb];
            for s in 0..ns {
                let d = digits[w * ns + s] as i32;
                if d == 0 {
                    continue;
                }
                let b = if d > 0 { (d - 1) as usize } else { (-d - 1) as usize };
                counts[b] += 1;
            }
            let mut start = vec![0u32; nb];
            let mut acc = 0u32;
            for b in 0..nb {
                start[b] = acc;
                acc += counts[b];
            }
            let mut cursor = start.clone();
            let mut idx = vec![0u32; ns];
            let mut sign = vec![0i8; ns];
            for s in 0..ns {
                let d = digits[w * ns + s] as i32;
                if d == 0 {
                    continue;
                }
                let b = if d > 0 { (d - 1) as usize } else { (-d - 1) as usize };
                let pos = cursor[b] as usize;
                cursor[b] += 1;
                idx[pos] = s as u32;
                sign[pos] = if d > 0 { 1 } else { -1 };
            }

            assert_eq!(&g_counts[w * nb..w * nb + nb], &counts[..], "counts w={w}");
            assert_eq!(&g_start[w * nb..w * nb + nb], &start[..], "start w={w}");
            // Only the first `acc` entries of the sorted arrays are defined.
            let placed = acc as usize;
            assert_eq!(&g_idx[w * ns..w * ns + placed], &idx[..placed], "idx w={w}");
            assert_eq!(&g_sign[w * ns..w * ns + placed], &sign[..placed], "sign w={w}");
        }
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
