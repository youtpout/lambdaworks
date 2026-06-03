//! Metal GPU-accelerated Pippenger MSM for 256-bit short-Weierstrass curves.
//!
//! Points are Jacobian `(x, y, z)` points with 4 little-endian `u64` limbs per
//! coordinate, stored in Montgomery form. The curve model is assumed to have
//! `a = 0`, which matches Pallas and Vesta.

use metal::Buffer;

use super::abstractions::{
    errors::{MetalError, MetalResult},
    state::DynamicMetalState,
};

const PIPPENGER_MSM_SHADER_SOURCE: &str = include_str!("shaders/pippenger_msm/pippenger_msm.metal");

const COORD_LIMBS: usize = 4;
const COORDS_PER_POINT: usize = 3;
const LIMBS_PER_POINT: usize = COORD_LIMBS * COORDS_PER_POINT;

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

/// Configuration for a 256-bit Pippenger MSM over an `a = 0` curve.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PippengerMSMConfig {
    /// Window size in bits.
    pub window_size: usize,
    /// Number of scalar limbs.
    pub scalar_limbs: usize,
    /// Bits per scalar limb.
    pub bits_per_limb: usize,
    /// Base-field modulus as little-endian limbs.
    pub field_modulus: [u64; COORD_LIMBS],
    /// Montgomery parameter `-p^{-1} mod 2^64`.
    pub montgomery_inv: u64,
}

impl PippengerMSMConfig {
    /// Pallas base-field configuration.
    pub fn pallas() -> Self {
        Self::for_modulus(PALLAS_MODULUS)
    }

    /// Vesta base-field configuration.
    pub fn vesta() -> Self {
        Self::for_modulus(VESTA_MODULUS)
    }

    /// Creates a config for any 256-bit `a = 0` curve whose coordinates use
    /// Montgomery representation over `field_modulus`.
    pub fn for_modulus(field_modulus: [u64; COORD_LIMBS]) -> Self {
        Self {
            window_size: 13,
            scalar_limbs: COORD_LIMBS,
            bits_per_limb: 64,
            montgomery_inv: montgomery_inv64(field_modulus[0]),
            field_modulus,
        }
    }

    /// Returns a practical window size for the input length.
    pub fn optimal_window_size(num_points: usize) -> usize {
        match num_points {
            0..=4 => 2,
            5..=32 => 4,
            33..=128 => 6,
            129..=1024 => 8,
            1025..=4096 => 10,
            4097..=16384 => 12,
            16385..=65536 => 13,
            _ => 14,
        }
    }

    pub const MAX_WINDOW_SIZE: usize = 20;

    fn validate(&self) {
        assert!(
            (1..=Self::MAX_WINDOW_SIZE).contains(&self.window_size),
            "window_size must be in 1..={}",
            Self::MAX_WINDOW_SIZE
        );
        assert_eq!(
            self.scalar_limbs, COORD_LIMBS,
            "only 256-bit scalars are supported"
        );
        assert_eq!(
            self.bits_per_limb, 64,
            "only 64-bit scalar limbs are supported"
        );
    }

    pub fn num_windows(&self) -> usize {
        self.validate();
        (self.scalar_limbs * self.bits_per_limb).div_ceil(self.window_size)
    }

    pub fn num_buckets(&self) -> usize {
        self.validate();
        1 << (self.window_size - 1)
    }
}

impl Default for PippengerMSMConfig {
    fn default() -> Self {
        Self::pallas()
    }
}

/// Metal Pippenger MSM.
pub struct MetalPippengerMSM {
    state: DynamicMetalState,
    config: PippengerMSMConfig,
    initialized: bool,
    max_threads_accumulation: u64,
    max_threads_reduction: u64,
}

impl MetalPippengerMSM {
    pub fn new(config: PippengerMSMConfig) -> MetalResult<Self> {
        let state = DynamicMetalState::new()?;
        Ok(Self {
            state,
            config,
            initialized: false,
            max_threads_accumulation: 0,
            max_threads_reduction: 0,
        })
    }

    pub fn new_pallas() -> MetalResult<Self> {
        Self::new(PippengerMSMConfig::pallas())
    }

    pub fn new_vesta() -> MetalResult<Self> {
        Self::new(PippengerMSMConfig::vesta())
    }

    pub fn initialize(&mut self) -> MetalResult<()> {
        if self.initialized {
            return Ok(());
        }

        self.state.load_library(PIPPENGER_MSM_SHADER_SOURCE)?;
        self.max_threads_accumulation = self
            .state
            .prepare_pipeline("bucket_accumulation_by_bucket")?;
        self.max_threads_reduction = self.state.prepare_pipeline("bucket_reduction")?;
        self.initialized = true;
        Ok(())
    }

    pub fn config(&self) -> &PippengerMSMConfig {
        &self.config
    }

    pub fn set_config(&mut self, config: PippengerMSMConfig) {
        self.config = config;
    }

    /// Computes `sum(scalars[i] * points[i])`.
    ///
    /// `scalars` is `[s0_l0, ..., s0_l3, s1_l0, ...]`.
    /// `points` is `[x0_l0..x0_l3, y0_l0..y0_l3, z0_l0..z0_l3, x1_l0..]`.
    pub fn compute(&mut self, scalars: &[u64], points: &[u64]) -> MetalResult<Vec<u64>> {
        if !self.initialized {
            self.initialize()?;
        }

        if !scalars.len().is_multiple_of(self.config.scalar_limbs) {
            return Err(MetalError::InvalidInputSize {
                expected: self.config.scalar_limbs,
                actual: scalars.len(),
            });
        }
        if !points.len().is_multiple_of(LIMBS_PER_POINT) {
            return Err(MetalError::InvalidInputSize {
                expected: LIMBS_PER_POINT,
                actual: points.len(),
            });
        }

        let num_scalars = scalars.len() / self.config.scalar_limbs;
        let num_points = points.len() / LIMBS_PER_POINT;
        if num_scalars != num_points {
            return Err(MetalError::LengthMismatch(num_scalars, num_points));
        }
        if num_scalars == 0 {
            return Err(MetalError::EmptyInput);
        }

        let signed_digits = recode_scalars_signed(&self.config, scalars, num_scalars);
        let digits_buffer = self.state.alloc_buffer_with_data(&signed_digits)?;
        let points_buffer = self.state.alloc_buffer_with_data(points)?;

        let num_buckets = self.config.num_buckets();
        let effective_windows = self.config.num_windows() + 1;
        let buckets_len = effective_windows * num_buckets * LIMBS_PER_POINT;
        let buckets_buffer = self
            .state
            .alloc_buffer(buckets_len * std::mem::size_of::<u64>())?;

        self.run_bucket_accumulation(
            &digits_buffer,
            &points_buffer,
            &buckets_buffer,
            num_scalars,
            effective_windows,
            num_buckets,
        )?;

        let window_sums_buffer = self
            .state
            .alloc_buffer(effective_windows * LIMBS_PER_POINT * std::mem::size_of::<u64>())?;

        self.run_bucket_reduction(
            &buckets_buffer,
            &window_sums_buffer,
            effective_windows,
            num_buckets,
        )?;

        let window_sums = unsafe {
            self.state
                .read_buffer::<u64>(&window_sums_buffer, effective_windows * LIMBS_PER_POINT)
        };

        Ok(self.combine_windows(&window_sums, effective_windows))
    }
}

fn recode_scalars_signed(
    config: &PippengerMSMConfig,
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
                    let remaining_bits = bit_in_limb + window_size - 64;
                    val |= (scalars[scalar_base + limb_idx + 1] & ((1u64 << remaining_bits) - 1))
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

impl MetalPippengerMSM {
    fn run_bucket_accumulation(
        &self,
        digits_buffer: &Buffer,
        points_buffer: &Buffer,
        buckets_buffer: &Buffer,
        num_scalars: usize,
        num_windows: usize,
        num_buckets: usize,
    ) -> MetalResult<()> {
        let config_data = [num_scalars as u32, num_windows as u32, num_buckets as u32];
        let config_buffer = self.state.alloc_buffer_with_data(&config_data)?;
        let field_buffer = self.field_params_buffer()?;
        let total_threads = (num_windows * num_buckets) as u64;

        self.state.execute_compute(
            "bucket_accumulation_by_bucket",
            &[
                digits_buffer,
                points_buffer,
                buckets_buffer,
                &config_buffer,
                &field_buffer,
            ],
            total_threads,
            self.max_threads_accumulation,
        )
    }

    fn run_bucket_reduction(
        &self,
        buckets_buffer: &Buffer,
        window_sums_buffer: &Buffer,
        num_windows: usize,
        num_buckets: usize,
    ) -> MetalResult<()> {
        let config_data = [num_windows as u32, num_buckets as u32];
        let config_buffer = self.state.alloc_buffer_with_data(&config_data)?;
        let field_buffer = self.field_params_buffer()?;

        self.state.execute_compute(
            "bucket_reduction",
            &[
                buckets_buffer,
                window_sums_buffer,
                &config_buffer,
                &field_buffer,
            ],
            num_windows as u64,
            self.max_threads_reduction,
        )
    }

    fn field_params_buffer(&self) -> MetalResult<Buffer> {
        let mut params = [0u64; COORD_LIMBS + 1];
        params[..COORD_LIMBS].copy_from_slice(&self.config.field_modulus);
        params[COORD_LIMBS] = self.config.montgomery_inv;
        self.state.alloc_buffer_with_data(&params)
    }

    fn combine_windows(&self, window_sums: &[u64], num_windows: usize) -> Vec<u64> {
        if num_windows == 0 {
            return vec![0; LIMBS_PER_POINT];
        }

        let field = FieldParams::from_config(&self.config);
        let mut result = JacobianPoint::from_limbs(
            &window_sums[(num_windows - 1) * LIMBS_PER_POINT..num_windows * LIMBS_PER_POINT],
        );

        for window_idx in (0..num_windows - 1).rev() {
            for _ in 0..self.config.window_size {
                result = result.double(&field);
            }
            let base = window_idx * LIMBS_PER_POINT;
            let window = JacobianPoint::from_limbs(&window_sums[base..base + LIMBS_PER_POINT]);
            result = result.add(&window, &field);
        }

        result.to_limbs()
    }
}

#[derive(Clone, Copy)]
struct FieldParams {
    modulus: [u64; COORD_LIMBS],
    inv: u64,
}

impl FieldParams {
    fn from_config(config: &PippengerMSMConfig) -> Self {
        Self {
            modulus: config.field_modulus,
            inv: config.montgomery_inv,
        }
    }
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
        let mut limbs = Vec::with_capacity(LIMBS_PER_POINT);
        limbs.extend_from_slice(&self.x);
        limbs.extend_from_slice(&self.y);
        limbs.extend_from_slice(&self.z);
        limbs
    }

    fn is_identity(&self) -> bool {
        self.z == [0; COORD_LIMBS]
    }

    fn double(&self, field: &FieldParams) -> Self {
        if self.is_identity() {
            return self.clone();
        }

        let a = mont_square(&self.x, field);
        let b = mont_square(&self.y, field);
        let c = mont_square(&b, field);
        let d = field_double(
            &field_sub(
                &field_sub(
                    &mont_square(&field_add(&self.x, &b, field), field),
                    &a,
                    field,
                ),
                &c,
                field,
            ),
            field,
        );
        let e = field_add(&a, &field_double(&a, field), field);
        let f = mont_square(&e, field);
        let x3 = field_sub(&f, &field_double(&d, field), field);
        let y3 = field_sub(
            &mont_mul(&e, &field_sub(&d, &x3, field), field),
            &field_double(&field_double(&field_double(&c, field), field), field),
            field,
        );
        let z3 = field_double(&mont_mul(&self.y, &self.z, field), field);

        Self {
            x: x3,
            y: y3,
            z: z3,
        }
    }

    fn add(&self, other: &Self, field: &FieldParams) -> Self {
        if self.is_identity() {
            return other.clone();
        }
        if other.is_identity() {
            return self.clone();
        }

        let z1z1 = mont_square(&self.z, field);
        let z2z2 = mont_square(&other.z, field);
        let u1 = mont_mul(&self.x, &z2z2, field);
        let u2 = mont_mul(&other.x, &z1z1, field);
        let s1 = mont_mul(&mont_mul(&self.y, &other.z, field), &z2z2, field);
        let s2 = mont_mul(&mont_mul(&other.y, &self.z, field), &z1z1, field);
        let h = field_sub(&u2, &u1, field);

        if h == [0; COORD_LIMBS] {
            if field_sub(&s2, &s1, field) == [0; COORD_LIMBS] {
                return self.double(field);
            }
            return Self::identity();
        }

        let i = mont_square(&field_double(&h, field), field);
        let j = mont_mul(&h, &i, field);
        let r = field_double(&field_sub(&s2, &s1, field), field);
        let v = mont_mul(&u1, &i, field);
        let x3 = field_sub(
            &field_sub(&mont_square(&r, field), &j, field),
            &field_double(&v, field),
            field,
        );
        let y3 = field_sub(
            &mont_mul(&r, &field_sub(&v, &x3, field), field),
            &field_double(&mont_mul(&s1, &j, field), field),
            field,
        );
        let z3 = mont_mul(
            &field_sub(
                &field_sub(
                    &mont_square(&field_add(&self.z, &other.z, field), field),
                    &z1z1,
                    field,
                ),
                &z2z2,
                field,
            ),
            &h,
            field,
        );

        Self {
            x: x3,
            y: y3,
            z: z3,
        }
    }
}

fn bigint_add(a: &[u64; COORD_LIMBS], b: &[u64; COORD_LIMBS]) -> ([u64; COORD_LIMBS], u64) {
    let mut result = [0; COORD_LIMBS];
    let mut carry = 0u64;
    for i in 0..COORD_LIMBS {
        let (sum1, c1) = a[i].overflowing_add(b[i]);
        let (sum2, c2) = sum1.overflowing_add(carry);
        result[i] = sum2;
        carry = u64::from(c1) + u64::from(c2);
    }
    (result, carry)
}

fn bigint_sub(a: &[u64; COORD_LIMBS], b: &[u64; COORD_LIMBS]) -> ([u64; COORD_LIMBS], u64) {
    let mut result = [0; COORD_LIMBS];
    let mut borrow = 0u64;
    for i in 0..COORD_LIMBS {
        let (diff1, b1) = a[i].overflowing_sub(b[i]);
        let (diff2, b2) = diff1.overflowing_sub(borrow);
        result[i] = diff2;
        borrow = u64::from(b1) + u64::from(b2);
    }
    (result, borrow)
}

fn field_add(
    a: &[u64; COORD_LIMBS],
    b: &[u64; COORD_LIMBS],
    field: &FieldParams,
) -> [u64; COORD_LIMBS] {
    let (sum, carry) = bigint_add(a, b);
    let (reduced, borrow) = bigint_sub(&sum, &field.modulus);
    if carry != 0 || borrow == 0 {
        reduced
    } else {
        sum
    }
}

fn field_sub(
    a: &[u64; COORD_LIMBS],
    b: &[u64; COORD_LIMBS],
    field: &FieldParams,
) -> [u64; COORD_LIMBS] {
    let (diff, borrow) = bigint_sub(a, b);
    if borrow != 0 {
        bigint_add(&diff, &field.modulus).0
    } else {
        diff
    }
}

fn field_double(a: &[u64; COORD_LIMBS], field: &FieldParams) -> [u64; COORD_LIMBS] {
    field_add(a, a, field)
}

fn mont_reduce(t: &[u64; COORD_LIMBS * 2], field: &FieldParams) -> [u64; COORD_LIMBS] {
    let mut tmp = *t;
    for i in 0..COORD_LIMBS {
        let m = tmp[i].wrapping_mul(field.inv);
        let mut carry = 0u64;
        for j in 0..COORD_LIMBS {
            let wide = (m as u128) * (field.modulus[j] as u128);
            let lo = wide as u64;
            let hi = (wide >> 64) as u64;
            let (sum1, c1) = tmp[i + j].overflowing_add(lo);
            let (sum2, c2) = sum1.overflowing_add(carry);
            tmp[i + j] = sum2;
            carry = hi + u64::from(c1) + u64::from(c2);
        }

        for j in COORD_LIMBS..(COORD_LIMBS * 2 - i) {
            let (sum, c) = tmp[i + j].overflowing_add(carry);
            tmp[i + j] = sum;
            carry = u64::from(c);
            if carry == 0 {
                break;
            }
        }
    }

    let mut result = [0; COORD_LIMBS];
    result.copy_from_slice(&tmp[COORD_LIMBS..]);
    let (reduced, borrow) = bigint_sub(&result, &field.modulus);
    if borrow == 0 {
        reduced
    } else {
        result
    }
}

fn mont_mul(
    a: &[u64; COORD_LIMBS],
    b: &[u64; COORD_LIMBS],
    field: &FieldParams,
) -> [u64; COORD_LIMBS] {
    let mut product = [0u64; COORD_LIMBS * 2];
    for i in 0..COORD_LIMBS {
        let mut carry = 0u64;
        for j in 0..COORD_LIMBS {
            let wide = (a[i] as u128) * (b[j] as u128);
            let lo = wide as u64;
            let hi = (wide >> 64) as u64;
            let (sum1, c1) = product[i + j].overflowing_add(lo);
            let (sum2, c2) = sum1.overflowing_add(carry);
            product[i + j] = sum2;
            carry = hi + u64::from(c1) + u64::from(c2);
        }
        let (sum, carry_overflow) = product[i + COORD_LIMBS].overflowing_add(carry);
        product[i + COORD_LIMBS] = sum;
        debug_assert!(!carry_overflow);
    }
    mont_reduce(&product, field)
}

fn mont_square(a: &[u64; COORD_LIMBS], field: &FieldParams) -> [u64; COORD_LIMBS] {
    mont_mul(a, a, field)
}

fn montgomery_inv64(modulus_limb: u64) -> u64 {
    let mut inv = 1u64;
    for _ in 0..6 {
        inv = inv.wrapping_mul(2u64.wrapping_sub(modulus_limb.wrapping_mul(inv)));
    }
    inv.wrapping_neg()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pallas_and_vesta_inverses_match_modulus_lsb() {
        for config in [PippengerMSMConfig::pallas(), PippengerMSMConfig::vesta()] {
            assert_eq!(
                config.field_modulus[0].wrapping_mul(config.montgomery_inv),
                u64::MAX
            );
        }
    }

    #[test]
    fn config_window_counts_are_256_bit() {
        let config = PippengerMSMConfig {
            window_size: 8,
            ..PippengerMSMConfig::pallas()
        };
        assert_eq!(config.num_windows(), 32);
        assert_eq!(config.num_buckets(), 128);
    }

    #[test]
    fn recoding_uses_extra_carry_window() {
        let config = PippengerMSMConfig {
            window_size: 4,
            ..PippengerMSMConfig::pallas()
        };

        let scalars = [u64::MAX; COORD_LIMBS];
        let digits = recode_scalars_signed(&config, &scalars, 1);
        assert_eq!(digits.len(), config.num_windows() + 1);
        assert_eq!(*digits.last().unwrap(), 1);
    }

    #[test]
    fn identity_arithmetic_is_stable() {
        let field = FieldParams::from_config(&PippengerMSMConfig::pallas());
        let id = JacobianPoint::identity();
        assert_eq!(id.double(&field), id);

        let point = JacobianPoint {
            x: [1, 2, 3, 4],
            y: [5, 6, 7, 8],
            z: [1, 0, 0, 0],
        };
        assert_eq!(id.add(&point, &field), point);
    }
}
