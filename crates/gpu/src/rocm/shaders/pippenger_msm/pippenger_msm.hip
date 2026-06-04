// HIP (ROCm) Pippenger MSM for 256-bit short-Weierstrass curves (a = 0).
//
// Points are Jacobian (x, y, z) in Montgomery form for bucket storage, but
// INPUT POINTS are affine (x, y only – 8 u64 limbs per point, not 12).
// This reduces H→D transfer and HBM read traffic by 33%.
//
// Kernels:
//   bucket_accumulation_by_chunk  – signed-digit bucket accumulation
//   bucket_merge                  – merge per-chunk partial buckets
//   bucket_reduction              – triangle sum reduction per window
//
// Design: one THREAD per (window, chunk) pair.  Each thread holds all buckets
// as private register variables.  To avoid VGPR spilling on RDNA3 (256 VGPR/lane
// limit), the caller must cap window_size ≤ 4 → num_buckets ≤ 8:
//   8 × 12 u64 = 192 VGPRs  <  256  →  zero spilling.
//
// Digits layout (transposed): [msm_idx][window_idx * num_scalars + point_idx]
// → for a fixed (window, chunk), consecutive gids read consecutive digit words
//   → coalesced L2 / L1 access.
//
// Batch MSM: blockIdx.y selects the MSM within a batch.
// Single-MSM callers set grid_dim_y = 1.

#define NUM_LIMBS 4
#define COORDS_PER_POINT 3
#define LIMBS_PER_POINT  (NUM_LIMBS * COORDS_PER_POINT)   // 12 – Jacobian bucket format
#define LIMBS_PER_AFFINE (NUM_LIMBS * 2)                  //  8 – affine input point format
#define MAX_PRIVATE_BUCKETS 8   // window_size=4 → num_buckets=8 → 192 VGPRs (fits RDNA3)

typedef unsigned long long u64limb;
typedef unsigned int       uint;

// ─── big-integer primitives ────────────────────────────────────────────────

struct BigInt      { u64limb limbs[NUM_LIMBS]; };
struct BigIntWide  { u64limb limbs[NUM_LIMBS * 2]; };

struct BigIntAddResult { BigInt result; u64limb carry; };
struct BigIntSubResult { BigInt result; u64limb borrow; };

struct FieldParams {
    BigInt   modulus;
    u64limb  inv;
    BigInt   mont_one;   // Montgomery representation of 1 (= R = 2^256 mod p)
};

struct JacobianPoint {
    BigInt x;
    BigInt y;
    BigInt z;
};

__device__ __forceinline__ u64limb mulhi_u64(u64limb a, u64limb b) {
    typedef unsigned __int128 u128;
    return (u64limb)(((u128)a * b) >> 64);
}

__device__ BigInt bigint_zero() {
    BigInt z;
    for (uint i = 0; i < NUM_LIMBS; i++) z.limbs[i] = 0;
    return z;
}

__device__ bool bigint_is_zero(BigInt a) {
    for (uint i = 0; i < NUM_LIMBS; i++) if (a.limbs[i] != 0) return false;
    return true;
}

__device__ BigIntAddResult bigint_add(BigInt a, BigInt b) {
    BigInt result;
    u64limb carry = 0;
    for (uint i = 0; i < NUM_LIMBS; i++) {
        u64limb s1 = a.limbs[i] + b.limbs[i];
        u64limb c1 = (s1 < a.limbs[i]) ? 1ULL : 0ULL;
        u64limb s2 = s1 + carry;
        u64limb c2 = (s2 < s1) ? 1ULL : 0ULL;
        result.limbs[i] = s2;
        carry = c1 + c2;
    }
    return {result, carry};
}

__device__ BigIntSubResult bigint_sub(BigInt a, BigInt b) {
    BigInt result;
    u64limb borrow = 0;
    for (uint i = 0; i < NUM_LIMBS; i++) {
        u64limb d1 = a.limbs[i] - b.limbs[i];
        u64limb b1 = (a.limbs[i] < b.limbs[i]) ? 1ULL : 0ULL;
        u64limb d2 = d1 - borrow;
        u64limb b2 = (d1 < borrow) ? 1ULL : 0ULL;
        result.limbs[i] = d2;
        borrow = b1 + b2;
    }
    return {result, borrow};
}

// field_params buffer layout: [modulus(4), inv(1), mont_one(4)] = 9 u64s
__device__ FieldParams load_field(const u64limb* __restrict__ params) {
    FieldParams f;
    for (uint i = 0; i < NUM_LIMBS; i++) f.modulus.limbs[i] = params[i];
    f.inv = params[NUM_LIMBS];
    for (uint i = 0; i < NUM_LIMBS; i++) f.mont_one.limbs[i] = params[NUM_LIMBS + 1 + i];
    return f;
}

// ─── Montgomery field arithmetic ─────────────────────────────────────────────

__device__ BigInt mont_mul(BigInt a, BigInt b, FieldParams f) {
    BigIntWide t;
    for (uint i = 0; i < NUM_LIMBS * 2; i++) t.limbs[i] = 0;

    for (uint i = 0; i < NUM_LIMBS; i++) {
        u64limb carry = 0;
        for (uint j = 0; j < NUM_LIMBS; j++) {
            u64limb lo  = a.limbs[i] * b.limbs[j];
            u64limb hi  = mulhi_u64(a.limbs[i], b.limbs[j]);
            u64limb s1  = t.limbs[i + j] + lo;
            u64limb c1  = (s1 < t.limbs[i + j]) ? 1ULL : 0ULL;
            u64limb s2  = s1 + carry;
            u64limb c2  = (s2 < s1) ? 1ULL : 0ULL;
            t.limbs[i + j] = s2;
            carry = hi + c1 + c2;
        }
        t.limbs[i + NUM_LIMBS] += carry;
    }

    for (uint i = 0; i < NUM_LIMBS; i++) {
        u64limb m = t.limbs[i] * f.inv;
        u64limb carry = 0;
        for (uint j = 0; j < NUM_LIMBS; j++) {
            u64limb lo  = m * f.modulus.limbs[j];
            u64limb hi  = mulhi_u64(m, f.modulus.limbs[j]);
            u64limb s1  = t.limbs[i + j] + lo;
            u64limb c1  = (s1 < t.limbs[i + j]) ? 1ULL : 0ULL;
            u64limb s2  = s1 + carry;
            u64limb c2  = (s2 < s1) ? 1ULL : 0ULL;
            t.limbs[i + j] = s2;
            carry = hi + c1 + c2;
        }
        u64limb s = t.limbs[i + NUM_LIMBS] + carry;
        u64limb c = (s < t.limbs[i + NUM_LIMBS]) ? 1ULL : 0ULL;
        t.limbs[i + NUM_LIMBS] = s;
        if (i + NUM_LIMBS + 1 < NUM_LIMBS * 2)
            t.limbs[i + NUM_LIMBS + 1] += c;
    }

    BigInt result;
    for (uint i = 0; i < NUM_LIMBS; i++) result.limbs[i] = t.limbs[i + NUM_LIMBS];

    // Conditional subtraction if result ≥ modulus.
    BigIntSubResult sub = bigint_sub(result, f.modulus);
    if (!sub.borrow) result = sub.result;
    return result;
}

__device__ BigInt mont_square(BigInt a, FieldParams f) {
    return mont_mul(a, a, f);
}

__device__ BigInt field_add(BigInt a, BigInt b, FieldParams f) {
    BigIntAddResult r = bigint_add(a, b);
    BigInt result = r.result;
    if (r.carry) {
        BigIntSubResult s = bigint_sub(result, f.modulus);
        return s.result;
    }
    BigIntSubResult s = bigint_sub(result, f.modulus);
    return s.borrow ? result : s.result;
}

__device__ BigInt field_double(BigInt a, FieldParams f) {
    return field_add(a, a, f);
}

__device__ BigInt field_sub(BigInt a, BigInt b, FieldParams f) {
    BigIntSubResult r = bigint_sub(a, b);
    if (r.borrow) {
        BigIntAddResult s = bigint_add(r.result, f.modulus);
        return s.result;
    }
    return r.result;
}

__device__ BigInt field_neg(BigInt a, FieldParams f) {
    if (bigint_is_zero(a)) return a;
    BigIntSubResult r = bigint_sub(f.modulus, a);
    return r.result;
}

// ─── Jacobian point operations ────────────────────────────────────────────────

__device__ JacobianPoint jacobian_identity() {
    JacobianPoint p;
    p.x = bigint_zero();
    p.y = bigint_zero();
    p.z = bigint_zero();
    return p;
}

__device__ bool jacobian_is_identity(JacobianPoint p) {
    return bigint_is_zero(p.z);
}

__device__ JacobianPoint jacobian_double(JacobianPoint p, FieldParams field) {
    if (jacobian_is_identity(p)) return p;
    BigInt A = mont_square(p.x, field);
    BigInt B = mont_square(p.y, field);
    BigInt C = mont_square(B, field);
    BigInt D = field_add(p.x, B, field);
    D = mont_square(D, field);
    D = field_sub(D, A, field);
    D = field_sub(D, C, field);
    D = field_double(D, field);
    BigInt E = field_add(A, field_double(A, field), field);
    BigInt F = mont_square(E, field);
    JacobianPoint result;
    result.x = field_sub(F, field_double(D, field), field);
    result.y = field_sub(
        mont_mul(E, field_sub(D, result.x, field), field),
        field_add(C, field_add(C, field_add(C, field_add(C, field_add(C, field_add(C, field_double(C, field), field), field), field), field), field), field),
        field);
    result.z = mont_mul(field_double(p.y, field), p.z, field);
    return result;
}

__device__ JacobianPoint jacobian_add(JacobianPoint p, JacobianPoint q, FieldParams field) {
    if (jacobian_is_identity(p)) return q;
    if (jacobian_is_identity(q)) return p;
    BigInt Z1Z1 = mont_square(p.z, field);
    BigInt Z2Z2 = mont_square(q.z, field);
    BigInt U1   = mont_mul(p.x, Z2Z2, field);
    BigInt U2   = mont_mul(q.x, Z1Z1, field);
    BigInt S1   = mont_mul(mont_mul(p.y, q.z, field), Z2Z2, field);
    BigInt S2   = mont_mul(mont_mul(q.y, p.z, field), Z1Z1, field);
    BigInt H    = field_sub(U2, U1, field);
    if (bigint_is_zero(H)) {
        if (bigint_is_zero(field_sub(S2, S1, field))) return jacobian_double(p, field);
        return jacobian_identity();
    }
    BigInt I   = mont_square(field_double(H, field), field);
    BigInt J   = mont_mul(H, I, field);
    BigInt r   = field_double(field_sub(S2, S1, field), field);
    BigInt V   = mont_mul(U1, I, field);
    JacobianPoint result;
    result.x = field_sub(field_sub(mont_square(r, field), J, field), field_double(V, field), field);
    result.y = field_sub(
        mont_mul(r, field_sub(V, result.x, field), field),
        field_double(mont_mul(S1, J, field), field),
        field);
    result.z = mont_mul(
        mont_mul(field_double(p.z, field), q.z, field),
        H, field);
    return result;
}

__device__ JacobianPoint jacobian_add_mixed(JacobianPoint p, JacobianPoint q, FieldParams field) {
    if (jacobian_is_identity(p)) return q;
    if (jacobian_is_identity(q)) return p;
    BigInt Z1Z1 = mont_square(p.z, field);
    BigInt U1   = p.x;
    BigInt U2   = mont_mul(q.x, Z1Z1, field);
    BigInt S1   = p.y;
    BigInt S2   = mont_mul(mont_mul(q.y, p.z, field), Z1Z1, field);
    BigInt H    = field_sub(U2, U1, field);
    if (bigint_is_zero(H)) {
        if (bigint_is_zero(field_sub(S2, S1, field))) return jacobian_double(p, field);
        return jacobian_identity();
    }
    BigInt HH  = mont_square(H, field);
    BigInt HHH = mont_mul(H, HH, field);
    BigInt r   = field_sub(S2, S1, field);
    BigInt V   = mont_mul(U1, HH, field);
    JacobianPoint result;
    result.x = field_sub(field_sub(mont_square(r, field), HHH, field), field_double(V, field), field);
    result.y = field_sub(
        mont_mul(r, field_sub(V, result.x, field), field),
        mont_mul(S1, HHH, field),
        field);
    result.z = mont_mul(p.z, H, field);
    return result;
}

__device__ JacobianPoint jacobian_neg(JacobianPoint p, FieldParams field) {
    p.y = field_neg(p.y, field);
    return p;
}

// ─── I/O helpers ─────────────────────────────────────────────────────────────

// Load an affine input point (x, y only — 8 limbs).  z is set to mont_one.
__device__ JacobianPoint load_affine_point(const u64limb* __restrict__ points, uint point_idx, BigInt mont_one) {
    uint base = point_idx * LIMBS_PER_AFFINE;
    JacobianPoint p;
    for (uint i = 0; i < NUM_LIMBS; i++) {
        p.x.limbs[i] = points[base + i];
        p.y.limbs[i] = points[base + NUM_LIMBS + i];
        p.z.limbs[i] = mont_one.limbs[i];
    }
    return p;
}

// Load a Jacobian bucket point (x, y, z — 12 limbs).
__device__ JacobianPoint load_jacobian_point(const u64limb* __restrict__ points, uint point_idx) {
    uint base = point_idx * LIMBS_PER_POINT;
    JacobianPoint p;
    for (uint i = 0; i < NUM_LIMBS; i++) {
        p.x.limbs[i] = points[base + i];
        p.y.limbs[i] = points[base + NUM_LIMBS + i];
        p.z.limbs[i] = points[base + NUM_LIMBS * 2 + i];
    }
    return p;
}

// Store a Jacobian point (12 limbs).
__device__ void store_jacobian_point(u64limb* __restrict__ points, uint point_idx, JacobianPoint p) {
    uint base = point_idx * LIMBS_PER_POINT;
    for (uint i = 0; i < NUM_LIMBS; i++) {
        points[base + i]                = p.x.limbs[i];
        points[base + NUM_LIMBS + i]    = p.y.limbs[i];
        points[base + NUM_LIMBS * 2 + i] = p.z.limbs[i];
    }
}

// ─── kernels ─────────────────────────────────────────────────────────────────
//
// blockIdx.y = MSM index within a batch.  Single-MSM callers set grid_dim_y = 1.

// Bucket accumulation.
//
// One THREAD per (window, chunk) pair.  Each thread accumulates all points in
// its chunk into a private array of num_buckets Jacobian points (registers).
//
// Caller must ensure window_size ≤ 4 → num_buckets ≤ MAX_PRIVATE_BUCKETS (8):
//   8 × 12 u64 = 192 VGPRs  <  256 VGPR/lane limit (RDNA3) → zero spilling.
//
// Digits: transposed layout [msm][window * num_scalars + scalar]
// → consecutive gids load consecutive digit words → coalesced L2/L1 access.
//
// config: [num_scalars, num_windows, num_buckets, chunk_size, num_chunks]
extern "C" __global__ void bucket_accumulation_by_chunk(
    const int*     __restrict__ digits,
    const u64limb* __restrict__ points,      // affine: LIMBS_PER_AFFINE per point
    u64limb*       __restrict__ partial_buckets,
    const uint*    __restrict__ config,
    const u64limb* __restrict__ field_params
) {
    uint gid         = blockIdx.x * blockDim.x + threadIdx.x;
    uint msm_idx     = blockIdx.y;
    uint num_scalars = config[0];
    uint num_windows = config[1];
    uint num_buckets = config[2];
    uint chunk_size  = config[3];
    uint num_chunks  = config[4];

    if (gid >= num_windows * num_chunks) return;

    uint window_idx = gid / num_chunks;
    uint chunk_idx  = gid % num_chunks;
    uint start      = chunk_idx * chunk_size;
    uint end        = (start + chunk_size < num_scalars) ? (start + chunk_size) : num_scalars;

    // Per-MSM offsets.
    // Digits: transposed layout [msm][window * num_scalars + scalar]
    uint msm_digits_off    = msm_idx * (num_windows * num_scalars);
    uint msm_points_off    = msm_idx * num_scalars;  // in affine-point units
    uint msm_partial_off   = msm_idx * (num_windows * num_chunks * num_buckets);
    uint partial_base      = msm_partial_off + (window_idx * num_chunks + chunk_idx) * num_buckets;
    uint window_digit_base = msm_digits_off + window_idx * num_scalars;

    FieldParams field = load_field(field_params);

    // Private register buckets.  num_buckets ≤ MAX_PRIVATE_BUCKETS = 8 → no VGPR spill.
    JacobianPoint local_buckets[MAX_PRIVATE_BUCKETS];
    for (uint b = 0; b < num_buckets; b++) local_buckets[b] = jacobian_identity();

    for (uint s = start; s < end; s++) {
        // Transposed layout → adjacent gids hit adjacent memory locations → coalesced.
        int digit = digits[window_digit_base + s];
        if (digit == 0) continue;
        uint bucket_idx = (digit > 0) ? (uint)(digit - 1) : (uint)(-digit - 1);
        JacobianPoint p = load_affine_point(points, msm_points_off + s, field.mont_one);
        if (digit < 0) p = jacobian_neg(p, field);
        local_buckets[bucket_idx] = jacobian_add_mixed(local_buckets[bucket_idx], p, field);
    }

    for (uint b = 0; b < num_buckets; b++)
        store_jacobian_point(partial_buckets, partial_base + b, local_buckets[b]);
}

extern "C" __global__ void bucket_merge(
    const u64limb* __restrict__ partial_buckets,
    u64limb*       __restrict__ buckets,
    const uint*    __restrict__ config,
    const u64limb* __restrict__ field_params
) {
    uint gid         = blockIdx.x * blockDim.x + threadIdx.x;
    uint msm_idx     = blockIdx.y;
    uint num_windows = config[0];
    uint num_buckets = config[1];
    uint num_chunks  = config[2];
    uint total_buckets = num_windows * num_buckets;

    if (gid >= total_buckets) return;

    uint window_idx = gid / num_buckets;
    uint bucket_idx = gid % num_buckets;

    uint msm_partial_off = msm_idx * (num_windows * num_chunks * num_buckets);
    uint msm_buckets_off = msm_idx * (num_windows * num_buckets);

    FieldParams field = load_field(field_params);
    JacobianPoint bucket = jacobian_identity();

    for (uint c = 0; c < num_chunks; c++) {
        uint partial_idx = msm_partial_off + (window_idx * num_chunks + c) * num_buckets + bucket_idx;
        JacobianPoint partial = load_jacobian_point(partial_buckets, partial_idx);
        bucket = jacobian_add(bucket, partial, field);
    }

    store_jacobian_point(buckets, msm_buckets_off + gid, bucket);
}

extern "C" __global__ void bucket_reduction(
    const u64limb* __restrict__ buckets,
    u64limb*       __restrict__ window_sums,
    const uint*    __restrict__ config,
    const u64limb* __restrict__ field_params
) {
    uint window_idx  = blockIdx.x * blockDim.x + threadIdx.x;
    uint msm_idx     = blockIdx.y;
    uint num_windows = config[0];
    uint num_buckets = config[1];

    if (window_idx >= num_windows) return;

    uint msm_buckets_off = msm_idx * (num_windows * num_buckets);
    uint msm_sums_off    = msm_idx * num_windows;

    FieldParams field = load_field(field_params);
    uint bucket_base = msm_buckets_off + window_idx * num_buckets;
    JacobianPoint running_sum = jacobian_identity();
    JacobianPoint result      = jacobian_identity();

    for (int i = (int)num_buckets - 1; i >= 0; i--) {
        JacobianPoint b = load_jacobian_point(buckets, bucket_base + (uint)i);
        running_sum = jacobian_add(running_sum, b, field);
        result      = jacobian_add(result, running_sum, field);
    }

    store_jacobian_point(window_sums, msm_sums_off + window_idx, result);
}
