// HIP (ROCm) Pippenger MSM for 256-bit short-Weierstrass curves (a = 0).
//
// Ported from the Metal implementation. Points are Jacobian (x, y, z) with
// 4 little-endian u64 limbs per coordinate in Montgomery form.
//
// Kernels:
//   clear_u64_buffer           – zero a u64 device buffer
//   bucket_accumulation_by_chunk – signed-digit bucket accumulation
//   bucket_merge               – merge per-chunk partial buckets
//   bucket_reduction           – triangle sum reduction per window
//
// Batch MSM: blockIdx.y selects the MSM within a batch.
// Single-MSM callers launch with grid_dim_y = 1 (blockIdx.y = 0, zero offset).
//
// Build with hipcc or consumed at runtime by hipRTC.

#define NUM_LIMBS 4
#define COORDS_PER_POINT 3
#define LIMBS_PER_POINT (NUM_LIMBS * COORDS_PER_POINT)
#define MAX_PRIVATE_BUCKETS 64

// hiprtc_runtime.h pre-defines u64limb as unsigned long (64-bit on Linux).
// We need unsigned long long for u64 limbs – use our own name.
typedef unsigned long long u64limb;
typedef unsigned int       uint;

// ─── big-integer primitives ────────────────────────────────────────────────

struct BigInt      { u64limb limbs[NUM_LIMBS]; };
struct BigIntWide  { u64limb limbs[NUM_LIMBS * 2]; };

struct BigIntAddResult { BigInt result; u64limb carry; };
struct BigIntSubResult { BigInt result; u64limb borrow; };

struct FieldParams {
    BigInt modulus;
    u64limb  inv;
};

struct JacobianPoint {
    BigInt x;
    BigInt y;
    BigInt z;
};

__device__ __forceinline__ u64limb mulhi_u64(u64limb a, u64limb b) {
    // __uint128_t is supported by both clang (HIP) and nvcc (CUDA).
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

__device__ FieldParams load_field(const u64limb* __restrict__ params) {
    FieldParams f;
    for (uint i = 0; i < NUM_LIMBS; i++) f.modulus.limbs[i] = params[i];
    f.inv = params[NUM_LIMBS];
    return f;
}

// ─── field arithmetic (Montgomery) ─────────────────────────────────────────

__device__ BigInt field_add(BigInt a, BigInt b, FieldParams field) {
    auto [sum, carry] = bigint_add(a, b);
    auto [reduced, borrow] = bigint_sub(sum, field.modulus);
    return (carry != 0 || borrow == 0) ? reduced : sum;
}

__device__ BigInt field_sub(BigInt a, BigInt b, FieldParams field) {
    auto [diff, borrow] = bigint_sub(a, b);
    if (borrow != 0) {
        auto [corrected, _] = bigint_add(diff, field.modulus);
        return corrected;
    }
    return diff;
}

__device__ BigInt field_neg(BigInt a, FieldParams field) {
    if (bigint_is_zero(a)) return a;
    auto [result, _] = bigint_sub(field.modulus, a);
    return result;
}

__device__ BigInt field_double(BigInt a, FieldParams field) {
    return field_add(a, a, field);
}

__device__ BigInt mont_reduce(BigIntWide t, FieldParams field) {
    BigIntWide tmp = t;
    for (uint i = 0; i < NUM_LIMBS; i++) {
        u64limb m = tmp.limbs[i] * field.inv;
        u64limb carry = 0;
        for (uint j = 0; j < NUM_LIMBS; j++) {
            u64limb lo = m * field.modulus.limbs[j];
            u64limb hi = mulhi_u64(m, field.modulus.limbs[j]);
            u64limb s1 = tmp.limbs[i + j] + lo;
            u64limb c1 = (s1 < tmp.limbs[i + j]) ? 1ULL : 0ULL;
            u64limb s2 = s1 + carry;
            u64limb c2 = (s2 < s1) ? 1ULL : 0ULL;
            tmp.limbs[i + j] = s2;
            carry = hi + c1 + c2;
        }
        for (uint j = NUM_LIMBS; i + j < NUM_LIMBS * 2; j++) {
            u64limb s = tmp.limbs[i + j] + carry;
            carry = (s < tmp.limbs[i + j]) ? 1ULL : 0ULL;
            tmp.limbs[i + j] = s;
            if (carry == 0) break;
        }
    }
    BigInt result;
    for (uint i = 0; i < NUM_LIMBS; i++) result.limbs[i] = tmp.limbs[i + NUM_LIMBS];
    auto [reduced, borrow] = bigint_sub(result, field.modulus);
    return (borrow == 0) ? reduced : result;
}

__device__ BigInt mont_mul(BigInt a, BigInt b, FieldParams field) {
    BigIntWide product;
    for (uint i = 0; i < NUM_LIMBS * 2; i++) product.limbs[i] = 0;
    for (uint i = 0; i < NUM_LIMBS; i++) {
        u64limb carry = 0;
        for (uint j = 0; j < NUM_LIMBS; j++) {
            u64limb lo = a.limbs[i] * b.limbs[j];
            u64limb hi = mulhi_u64(a.limbs[i], b.limbs[j]);
            u64limb s1 = product.limbs[i + j] + lo;
            u64limb c1 = (s1 < product.limbs[i + j]) ? 1ULL : 0ULL;
            u64limb s2 = s1 + carry;
            u64limb c2 = (s2 < s1) ? 1ULL : 0ULL;
            product.limbs[i + j] = s2;
            carry = hi + c1 + c2;
        }
        product.limbs[i + NUM_LIMBS] += carry;
    }
    return mont_reduce(product, field);
}

__device__ BigInt mont_square(BigInt a, FieldParams field) {
    return mont_mul(a, a, field);
}

// ─── Jacobian point operations ──────────────────────────────────────────────

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
        field_double(field_double(field_double(C, field), field), field),
        field);
    result.z = field_double(mont_mul(p.y, p.z, field), field);
    return result;
}

__device__ JacobianPoint jacobian_add(JacobianPoint p, JacobianPoint q, FieldParams field) {
    if (jacobian_is_identity(p)) return q;
    if (jacobian_is_identity(q)) return p;
    BigInt Z1Z1 = mont_square(p.z, field);
    BigInt Z2Z2 = mont_square(q.z, field);
    BigInt U1 = mont_mul(p.x, Z2Z2, field);
    BigInt U2 = mont_mul(q.x, Z1Z1, field);
    BigInt S1 = mont_mul(mont_mul(p.y, q.z, field), Z2Z2, field);
    BigInt S2 = mont_mul(mont_mul(q.y, p.z, field), Z1Z1, field);
    BigInt H  = field_sub(U2, U1, field);
    if (bigint_is_zero(H)) {
        if (bigint_is_zero(field_sub(S2, S1, field))) return jacobian_double(p, field);
        return jacobian_identity();
    }
    BigInt I = mont_square(field_double(H, field), field);
    BigInt J = mont_mul(H, I, field);
    BigInt r = field_double(field_sub(S2, S1, field), field);
    BigInt V = mont_mul(U1, I, field);
    JacobianPoint result;
    result.x = field_sub(field_sub(mont_square(r, field), J, field), field_double(V, field), field);
    result.y = field_sub(
        mont_mul(r, field_sub(V, result.x, field), field),
        field_double(mont_mul(S1, J, field), field),
        field);
    result.z = mont_mul(
        field_sub(field_sub(mont_square(field_add(p.z, q.z, field), field), Z1Z1, field), Z2Z2, field),
        H, field);
    return result;
}

// Mixed addition: p is Jacobian, q is affine (z = R, the Montgomery one).
// Saves ~5 field multiplications vs full jacobian_add.
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

__device__ JacobianPoint load_point(const u64limb* __restrict__ points, uint point_idx) {
    uint base = point_idx * LIMBS_PER_POINT;
    JacobianPoint p;
    for (uint i = 0; i < NUM_LIMBS; i++) {
        p.x.limbs[i] = points[base + i];
        p.y.limbs[i] = points[base + NUM_LIMBS + i];
        p.z.limbs[i] = points[base + 2 * NUM_LIMBS + i];
    }
    return p;
}

__device__ void store_point(u64limb* points, uint point_idx, JacobianPoint p) {
    uint base = point_idx * LIMBS_PER_POINT;
    for (uint i = 0; i < NUM_LIMBS; i++) {
        points[base + i]                 = p.x.limbs[i];
        points[base + NUM_LIMBS + i]     = p.y.limbs[i];
        points[base + 2 * NUM_LIMBS + i] = p.z.limbs[i];
    }
}

// ─── kernels ─────────────────────────────────────────────────────────────────
//
// blockIdx.y = MSM index within a batch.  Single-MSM callers set grid_dim_y=1.

extern "C" __global__ void clear_u64_buffer(
    u64limb*      __restrict__ buffer,
    const uint*   __restrict__ config
) {
    uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint len = config[0];
    if (gid < len) buffer[gid] = 0;
}

extern "C" __global__ void bucket_accumulation_by_chunk(
    const int*     __restrict__ digits,
    const u64limb* __restrict__ points,
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
    uint total_workers = num_windows * num_chunks;

    if (gid >= total_workers) return;

    uint window_idx = gid / num_chunks;
    uint chunk_idx  = gid % num_chunks;
    uint start      = chunk_idx * chunk_size;
    uint end        = (start + chunk_size < num_scalars) ? (start + chunk_size) : num_scalars;

    // Per-MSM offsets into the concatenated buffers.
    uint msm_digits_off   = msm_idx * (num_scalars * num_windows);
    uint msm_points_off   = msm_idx * (num_scalars);            // in point units
    uint msm_partial_off  = msm_idx * (num_windows * num_chunks * num_buckets); // in point units
    uint partial_base     = msm_partial_off + (window_idx * num_chunks + chunk_idx) * num_buckets;

    FieldParams field = load_field(field_params);

    if (num_buckets <= MAX_PRIVATE_BUCKETS) {
        JacobianPoint local_buckets[MAX_PRIVATE_BUCKETS];
        for (uint b = 0; b < num_buckets; b++) local_buckets[b] = jacobian_identity();

        for (uint s = start; s < end; s++) {
            int digit = digits[msm_digits_off + s * num_windows + window_idx];
            if (digit == 0) continue;
            uint bucket_idx = (digit > 0) ? (uint)(digit - 1) : (uint)(-digit - 1);
            JacobianPoint p = load_point(points, msm_points_off + s);
            if (digit < 0) p = jacobian_neg(p, field);
            local_buckets[bucket_idx] = jacobian_add_mixed(local_buckets[bucket_idx], p, field);
        }

        for (uint b = 0; b < num_buckets; b++)
            store_point(partial_buckets, partial_base + b, local_buckets[b]);
        return;
    }

    // Slow path: buckets in global memory.
    for (uint s = start; s < end; s++) {
        int digit = digits[msm_digits_off + s * num_windows + window_idx];
        if (digit == 0) continue;
        uint bucket_idx = (digit > 0) ? (uint)(digit - 1) : (uint)(-digit - 1);
        JacobianPoint p = load_point(points, msm_points_off + s);
        if (digit < 0) p = jacobian_neg(p, field);
        uint partial_idx = partial_base + bucket_idx;
        JacobianPoint bucket = load_point(partial_buckets, partial_idx);
        bucket = jacobian_add_mixed(bucket, p, field);
        store_point(partial_buckets, partial_idx, bucket);
    }
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
        JacobianPoint partial = load_point(partial_buckets, partial_idx);
        bucket = jacobian_add(bucket, partial, field);
    }

    store_point(buckets, msm_buckets_off + gid, bucket);
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
        JacobianPoint b = load_point(buckets, bucket_base + (uint)i);
        running_sum = jacobian_add(running_sum, b, field);
        result      = jacobian_add(result, running_sum, field);
    }

    store_point(window_sums, msm_sums_off + window_idx, result);
}
