// HIP (ROCm) Pippenger MSM for 256-bit short-Weierstrass curves (a = 0).
//
// Compile with -DPALLAS_CURVE or -DVESTA_CURVE to activate the specialised
// Montgomery multiply for those moduli.  Without either flag the code falls
// back to the fully generic path.
//
// Specialisation targets the Pallas / Vesta modulus structure:
//   modulus = [m0, m1, 0, 2^62]
// This eliminates one multiply per reduction step (j=2, modulus=0) and
// replaces another with a shift pair (j=3, modulus=2^62).  ~25 % fewer
// multiplies in mont_mul → measurable throughput gain in bucket_accumulation.
//
// Kernels:
//   bucket_accumulation_by_chunk – signed-digit bucket accumulation
//   bucket_merge                 – parallel LDS merge of per-chunk partials
//   bucket_reduce_combine        – fused triangle-sum + Horner window combine
//
// Design: one THREAD per (window, chunk) pair.  Each thread holds all buckets
// as private register variables.  window_size ≤ 4 → num_buckets ≤ 8 → 192
// VGPRs per thread < 256 VGPR/lane limit (RDNA3) → zero register spilling.
//
// Digits layout (transposed): [msm_idx][window_idx * num_scalars + point_idx]
// → consecutive gids load consecutive digit words → coalesced L2/L1 access.
//
// Batch MSM: blockIdx.y selects the MSM.  Single-MSM callers set grid_dim_y=1.

// ─── per-curve compile-time constants ────────────────────────────────────────
//
// Pallas:  p = 0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001
// Vesta:   p = 0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001
//
// Both share the property: limbs[2]=0, limbs[3]=2^62.
// Compile with -DPALLAS_CURVE or -DVESTA_CURVE to select constants.

// Modulus and Montgomery inverse constants (per curve).
// mont_one (R = 2^256 mod p) is still loaded from the field_params buffer —
// not needed at compile time.
#ifdef PALLAS_CURVE
#define CURVE_MOD0     0x992d30ed00000001ULL
#define CURVE_MOD1     0x224698fc094cf91bULL
#define CURVE_MONT_INV 0x992d30ecffffffffULL
#endif

#ifdef VESTA_CURVE
#define CURVE_MOD0     0x8c46eb2100000001ULL
#define CURVE_MOD1     0x224698fc0994a8ddULL
#define CURVE_MONT_INV 0x8c46eb20ffffffffULL
#endif

#define NUM_LIMBS 4
#define COORDS_PER_POINT 3
#define LIMBS_PER_POINT  (NUM_LIMBS * COORDS_PER_POINT)   // 12 – Jacobian bucket format
#define LIMBS_PER_AFFINE (NUM_LIMBS * 2)                  //  8 – affine input point format
#define MAX_PRIVATE_BUCKETS 8   // window_size=4 → num_buckets=8 → 192 VGPRs (fits RDNA3)
#define MERGE_BLOCK_SIZE    64  // threads per block for bucket_merge LDS tree reduction
                                // LDS per block: 64 × 12 × 8 = 6 144 B  (well under 64 KB/CU)

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
//
// For Pallas / Vesta the modulus and inv are compile-time constants; only
// mont_one is read from the buffer, saving 5 global-memory loads per thread.
__device__ FieldParams load_field(const u64limb* __restrict__ params) {
    FieldParams f;
#if defined(PALLAS_CURVE) || defined(VESTA_CURVE)
    f.modulus.limbs[0] = CURVE_MOD0;
    f.modulus.limbs[1] = CURVE_MOD1;
    f.modulus.limbs[2] = 0ULL;
    f.modulus.limbs[3] = 0x4000000000000000ULL;
    f.inv = CURVE_MONT_INV;
#else
    for (uint i = 0; i < NUM_LIMBS; i++) f.modulus.limbs[i] = params[i];
    f.inv = params[NUM_LIMBS];
#endif
    for (uint i = 0; i < NUM_LIMBS; i++) f.mont_one.limbs[i] = params[NUM_LIMBS + 1 + i];
    return f;
}

// ─── Montgomery field arithmetic ─────────────────────────────────────────────

// Inline helper: add (lo,hi) of a 64×64 product into t[pos] with carry in/out.
#define MULADD(t_pos, a_val, b_val, carry)                        \
    do {                                                           \
        u64limb _lo = (a_val) * (b_val);                          \
        u64limb _hi = mulhi_u64((a_val), (b_val));                \
        u64limb _s1 = (t_pos) + _lo;                              \
        u64limb _c1 = (_s1 < (t_pos)) ? 1ULL : 0ULL;             \
        u64limb _s2 = _s1 + (carry);                              \
        u64limb _c2 = (_s2 < _s1) ? 1ULL : 0ULL;                 \
        (t_pos) = _s2;                                             \
        (carry) = _hi + _c1 + _c2;                                \
    } while(0)

// Inline helper: add a shift-based product  m × 2^62  into t[pos],t[pos+1].
// m × 2^62: lo = m<<62, hi = m>>2.
#define MULADD_2_62(t_lo, t_hi, m_val, carry)                    \
    do {                                                           \
        u64limb _lo = (m_val) << 62;                              \
        u64limb _hi = (m_val) >> 2;                               \
        u64limb _s1 = (t_lo) + _lo;                               \
        u64limb _c1 = (_s1 < (t_lo)) ? 1ULL : 0ULL;              \
        u64limb _s2 = _s1 + (carry);                              \
        u64limb _c2 = (_s2 < _s1) ? 1ULL : 0ULL;                 \
        (t_lo) = _s2;                                              \
        (carry) = _hi + _c1 + _c2;                                \
        (void)(t_hi); /* caller propagates carry */                \
    } while(0)

__device__ BigInt mont_mul(BigInt a, BigInt b, FieldParams f) {
    BigIntWide t;
    for (uint i = 0; i < NUM_LIMBS * 2; i++) t.limbs[i] = 0;

    // ── Step 1: compute full 512-bit product a × b ───────────────────────────
    for (uint i = 0; i < NUM_LIMBS; i++) {
        u64limb carry = 0;
        for (uint j = 0; j < NUM_LIMBS; j++) {
            MULADD(t.limbs[i + j], a.limbs[i], b.limbs[j], carry);
        }
        t.limbs[i + NUM_LIMBS] += carry;
    }

    // ── Step 2: Montgomery reduction ─────────────────────────────────────────
#if defined(PALLAS_CURVE) || defined(VESTA_CURVE)
    // Specialised for modulus = [CURVE_MOD0, CURVE_MOD1, 0, 2^62].
    //   j=2: modulus[2]=0  → no multiply, just carry propagation
    //   j=3: modulus[3]=2^62 → replace multiply with two shifts
    for (uint i = 0; i < NUM_LIMBS; i++) {
        u64limb m     = t.limbs[i] * CURVE_MONT_INV;
        u64limb carry = 0;
        // j=0
        MULADD(t.limbs[i    ], m, CURVE_MOD0, carry);
        // j=1
        MULADD(t.limbs[i + 1], m, CURVE_MOD1, carry);
        // j=2: modulus[2]=0 — skip multiply, propagate carry only
        { u64limb s = t.limbs[i + 2] + carry;
          carry = (s < t.limbs[i + 2]) ? 1ULL : 0ULL;
          t.limbs[i + 2] = s; }
        // j=3: modulus[3]=2^62 — use shifts
        MULADD_2_62(t.limbs[i + 3], t.limbs[i + 4], m, carry);
        // Propagate final carry into the upper half.
        u64limb s = t.limbs[i + NUM_LIMBS] + carry;
        u64limb c = (s < t.limbs[i + NUM_LIMBS]) ? 1ULL : 0ULL;
        t.limbs[i + NUM_LIMBS] = s;
        if (i + NUM_LIMBS + 1 < NUM_LIMBS * 2)
            t.limbs[i + NUM_LIMBS + 1] += c;
    }
#else
    // Generic reduction using dynamic field params.
    for (uint i = 0; i < NUM_LIMBS; i++) {
        u64limb m = t.limbs[i] * f.inv;
        u64limb carry = 0;
        for (uint j = 0; j < NUM_LIMBS; j++) {
            MULADD(t.limbs[i + j], m, f.modulus.limbs[j], carry);
        }
        u64limb s = t.limbs[i + NUM_LIMBS] + carry;
        u64limb c = (s < t.limbs[i + NUM_LIMBS]) ? 1ULL : 0ULL;
        t.limbs[i + NUM_LIMBS] = s;
        if (i + NUM_LIMBS + 1 < NUM_LIMBS * 2)
            t.limbs[i + NUM_LIMBS + 1] += c;
    }
#endif

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

// ─── Montgomery modular inverse (Fermat: a^(p-2)) ─────────────────────────────
//
// Inputs/outputs in Montgomery form.  ~254 squarings + ~one mul per set exponent
// bit ≈ 380 mont_muls.  Expensive on its own — only used as the single "real"
// inversion that the batch-inversion trick amortises over many elements.
__device__ BigInt mont_inverse(BigInt a, FieldParams f) {
    // exponent = modulus - 2
    BigInt two = bigint_zero(); two.limbs[0] = 2ULL;
    BigInt exp = bigint_sub(f.modulus, two).result;

    BigInt result = f.mont_one;   // 1 in Montgomery form
    BigInt base   = a;
    for (uint limb = 0; limb < NUM_LIMBS; limb++) {
        u64limb e = exp.limbs[limb];
        for (uint bit = 0; bit < 64; bit++) {
            if (e & 1ULL) result = mont_mul(result, base, f);
            base = mont_square(base, f);
            e >>= 1;
        }
    }
    return result;
}

// Batch inversion via Montgomery's trick over a contiguous array slice held by
// one thread.  N independent inverses cost 3·(N−1) muls + 1 inversion instead of
// N inversions.  `out` must NOT alias `vals` (the back loop reads vals[i] after
// writing out[i]).  Caller guarantees all vals ≠ 0.
//
//   prefix products → single inverse → back-substitution
__device__ void batch_inverse_local(const BigInt* vals, BigInt* out, uint n, FieldParams f,
                                     BigInt* scratch) {
    // scratch[i] = vals[0] * vals[1] * ... * vals[i]
    scratch[0] = vals[0];
    for (uint i = 1; i < n; i++) scratch[i] = mont_mul(scratch[i - 1], vals[i], f);

    BigInt inv = mont_inverse(scratch[n - 1], f);   // (∏ vals)^{-1}

    for (uint i = n - 1; i > 0; i--) {
        out[i] = mont_mul(inv, scratch[i - 1], f);   // vals[i]^{-1}
        inv    = mont_mul(inv, vals[i], f);          // strip vals[i]
    }
    out[0] = inv;
}

// ─── affine point batch addition ──────────────────────────────────────────────

struct AffinePoint { BigInt x; BigInt y; };

// Add n independent pairs R[i] = P[i] + Q[i] in affine coordinates, sharing a
// single field inversion across all pairs (Montgomery's trick).
//
// Per pair (distinct, P[i] ≠ ±Q[i]):
//   λ  = (yQ − yP) / (xQ − xP)
//   xR = λ² − xP − xQ
//   yR = λ·(xP − xR) − yP
//
// Cost ≈ 6 muls/pair + 1 amortised inversion (≈ 380/n muls).  For n ≫ 64 this is
// far cheaper than the ~10 mont_muls of a Jacobian mixed-add.
//
// Caller guarantees xP[i] ≠ xQ[i] for all i (no doubling / identity in batch).
// `denom`, `denom_inv`, `scratch` are caller-provided length-n workspaces.
__device__ void batch_affine_add_local(
    const AffinePoint* P, const AffinePoint* Q, AffinePoint* R, uint n, FieldParams f,
    BigInt* denom, BigInt* denom_inv, BigInt* scratch
) {
    for (uint i = 0; i < n; i++) denom[i] = field_sub(Q[i].x, P[i].x, f);
    batch_inverse_local(denom, denom_inv, n, f, scratch);
    for (uint i = 0; i < n; i++) {
        BigInt num    = field_sub(Q[i].y, P[i].y, f);
        BigInt lambda = mont_mul(num, denom_inv[i], f);
        BigInt l2     = mont_square(lambda, f);
        BigInt xr     = field_sub(field_sub(l2, P[i].x, f), Q[i].x, f);
        BigInt yr     = field_sub(mont_mul(lambda, field_sub(P[i].x, xr, f), f), P[i].y, f);
        R[i].x = xr;
        R[i].y = yr;
    }
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
    const signed char* __restrict__ digits,  // i8: fits window_size ≤ 7 (range −64..64)
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
        // Cast to int to preserve sign extension from the i8 representation.
        int digit = (int)digits[window_digit_base + s];
        if (digit == 0) continue;
        uint bucket_idx = (digit > 0) ? (uint)(digit - 1) : (uint)(-digit - 1);
        JacobianPoint p = load_affine_point(points, msm_points_off + s, field.mont_one);
        if (digit < 0) p = jacobian_neg(p, field);
        local_buckets[bucket_idx] = jacobian_add_mixed(local_buckets[bucket_idx], p, field);
    }

    for (uint b = 0; b < num_buckets; b++)
        store_jacobian_point(partial_buckets, partial_base + b, local_buckets[b]);
}

// Merge per-chunk partial buckets into one bucket per (window, bucket) pair.
//
// Launch geometry:
//   blockIdx.x  = window_idx * num_buckets + bucket_idx  (one block per pair)
//   blockIdx.y  = MSM index within a batch
//   threadIdx.x = lane within block  (0 .. MERGE_BLOCK_SIZE-1)
//
// Each thread strides over chunks, accumulates a local sub-sum, then the block
// performs a tree reduction in LDS to produce the final merged bucket.
//
// With MERGE_BLOCK_SIZE = 64 = one full wavefront (RDNA3), __syncthreads() is a
// no-op; the tree reduction is therefore free of cross-wavefront synchronisation.
// LDS: 64 × 12 × 8 = 6 144 bytes/block → fits easily in the 64 KB/CU budget.
//
// config: [num_windows, num_buckets, num_chunks]
extern "C" __global__ void bucket_merge(
    const u64limb* __restrict__ partial_buckets,
    u64limb*       __restrict__ buckets,
    const uint*    __restrict__ config,
    const u64limb* __restrict__ field_params
) {
    uint pair_idx    = blockIdx.x;   // linear (window, bucket) index
    uint msm_idx     = blockIdx.y;
    uint num_windows = config[0];
    uint num_buckets = config[1];
    uint num_chunks  = config[2];
    uint tid         = threadIdx.x;  // 0 .. MERGE_BLOCK_SIZE-1

    if (pair_idx >= num_windows * num_buckets) return;

    uint window_idx = pair_idx / num_buckets;
    uint bucket_idx = pair_idx % num_buckets;

    uint msm_partial_off = msm_idx * (num_windows * num_chunks * num_buckets);
    FieldParams field = load_field(field_params);

    // ── Phase 1: each thread strides over its chunk slice ────────────────────
    JacobianPoint local = jacobian_identity();
    for (uint c = tid; c < num_chunks; c += MERGE_BLOCK_SIZE) {
        uint partial_idx = msm_partial_off
                         + (window_idx * num_chunks + c) * num_buckets
                         + bucket_idx;
        JacobianPoint partial = load_jacobian_point(partial_buckets, partial_idx);
        local = jacobian_add(local, partial, field);
    }

    // ── Phase 2: write local sub-sums into LDS ───────────────────────────────
    __shared__ u64limb lds[MERGE_BLOCK_SIZE * LIMBS_PER_POINT];
    store_jacobian_point(lds, tid, local);
    __syncthreads();

    // ── Phase 3: tree reduction in LDS (log2(MERGE_BLOCK_SIZE) = 6 steps) ───
    for (uint stride = MERGE_BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            JacobianPoint a = load_jacobian_point(lds, tid);
            JacobianPoint b = load_jacobian_point(lds, tid + stride);
            store_jacobian_point(lds, tid, jacobian_add(a, b, field));
        }
        __syncthreads();
    }

    // ── Phase 4: lane 0 writes the merged bucket ──────────────────────────────
    if (tid == 0) {
        uint msm_buckets_off = msm_idx * (num_windows * num_buckets);
        store_jacobian_point(buckets, msm_buckets_off + pair_idx,
                             load_jacobian_point(lds, 0));
    }
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

// Fused triangle-reduction + Horner window combination.
//
// Replaces the separate bucket_reduction + combine_windows_kernel pair.
// Savings vs the old design:
//   • One fewer kernel launch and hipDeviceSynchronize barrier.
//   • The window_sums intermediate buffer is never written to global memory;
//     window results pass through LDS within the block.
//   • The final combined point (12 u64) goes directly to the result buffer.
//
// Launch geometry:
//   blockIdx.y  = MSM index within the batch  (or 0 for single-MSM)
//   blockIdx.x  = 0  (one block per MSM — pass grid_x=1, grid_y=num_msms)
//   threadIdx.x = window index  (0 .. num_windows-1)
//   blockDim.x  = num_windows   (dynamic, set by host)
//
// Shared memory (dynamic): num_windows × LIMBS_PER_POINT × 8 bytes.
//   For window_size=4 and 256-bit scalars: 65 × 12 × 8 = 6 240 bytes.
//
// config: [num_windows, num_buckets, window_size]
extern "C" __global__ void bucket_reduce_combine(
    const u64limb* __restrict__ buckets,   // [num_msms × num_windows × num_buckets × 12]
    u64limb*       __restrict__ results,   // [num_msms × 12]
    const uint*    __restrict__ config,
    const u64limb* __restrict__ field_params
) {
    uint msm_idx     = blockIdx.y;
    uint window_idx  = threadIdx.x;
    uint num_windows = config[0];
    uint num_buckets = config[1];
    uint window_size = config[2];

    extern __shared__ u64limb lds_rc[];  // dynamic: num_windows × LIMBS_PER_POINT u64limbs

    FieldParams field = load_field(field_params);

    // ── Phase 1: triangle-sum reduction for window window_idx ────────────────
    uint bucket_base = (msm_idx * num_windows + window_idx) * num_buckets;
    JacobianPoint running = jacobian_identity();
    JacobianPoint window_sum = jacobian_identity();

    for (int i = (int)num_buckets - 1; i >= 0; i--) {
        JacobianPoint b = load_jacobian_point(buckets, bucket_base + (uint)i);
        running     = jacobian_add(running,    b,       field);
        window_sum  = jacobian_add(window_sum, running, field);
    }

    // ── Phase 2: store window_sum to LDS ────────────────────────────────────
    store_jacobian_point(lds_rc, window_idx, window_sum);
    __syncthreads();

    // ── Phase 3: Horner combination in thread 0 only ─────────────────────────
    if (window_idx == 0) {
        JacobianPoint result = load_jacobian_point(lds_rc, num_windows - 1);
        for (int i = (int)num_windows - 2; i >= 0; i--) {
            for (uint d = 0; d < window_size; d++)
                result = jacobian_double(result, field);
            JacobianPoint w = load_jacobian_point(lds_rc, (uint)i);
            result = jacobian_add(result, w, field);
        }
        store_jacobian_point(results, msm_idx, result);
    }
}

// Combine per-window sums into the final MSM result (Horner's method).
//
// One thread per MSM.  Sequential but saves the PCIe readback of all window
// sums — only the final point (12 u64) is returned to the host.
//
// config: [num_windows, window_size, num_msms]
extern "C" __global__ void combine_windows_kernel(
    const u64limb* __restrict__ window_sums,  // [num_msms × num_windows × LIMBS_PER_POINT]
    u64limb*       __restrict__ results,       // [num_msms × LIMBS_PER_POINT]
    const uint*    __restrict__ config,
    const u64limb* __restrict__ field_params
) {
    uint msm_idx     = blockIdx.x * blockDim.x + threadIdx.x;
    uint num_windows = config[0];
    uint window_size = config[1];
    uint num_msms    = config[2];

    if (msm_idx >= num_msms) return;

    FieldParams field = load_field(field_params);
    uint sums_base = msm_idx * num_windows;

    // Horner from the most-significant window down.
    JacobianPoint result = load_jacobian_point(window_sums, sums_base + num_windows - 1);
    for (int i = (int)num_windows - 2; i >= 0; i--) {
        for (uint d = 0; d < window_size; d++)
            result = jacobian_double(result, field);
        JacobianPoint w = load_jacobian_point(window_sums, sums_base + (uint)i);
        result = jacobian_add(result, w, field);
    }

    store_jacobian_point(results, msm_idx, result);
}

// ─── test kernel: batch inverse ──────────────────────────────────────────────
//
// Single thread inverts `n` field elements (Montgomery form) in place.
// config: [n].  Used by the correctness test only.
#define BATCH_INV_TEST_MAX 512
extern "C" __global__ void batch_inverse_test(
    const u64limb* __restrict__ vals,   // [n × NUM_LIMBS]
    u64limb*       __restrict__ out,    // [n × NUM_LIMBS]
    const uint*    __restrict__ config,
    const u64limb* __restrict__ field_params
) {
    if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;
    uint n = config[0];
    if (n > BATCH_INV_TEST_MAX) return;

    FieldParams field = load_field(field_params);

    BigInt in[BATCH_INV_TEST_MAX];
    BigInt res[BATCH_INV_TEST_MAX];
    BigInt scratch[BATCH_INV_TEST_MAX];
    for (uint i = 0; i < n; i++)
        for (uint l = 0; l < NUM_LIMBS; l++)
            in[i].limbs[l] = vals[i * NUM_LIMBS + l];

    batch_inverse_local(in, res, n, field, scratch);

    for (uint i = 0; i < n; i++)
        for (uint l = 0; l < NUM_LIMBS; l++)
            out[i * NUM_LIMBS + l] = res[i].limbs[l];
}

// ─── test kernel: batch affine addition ──────────────────────────────────────
//
// Single thread computes R[i] = P[i] + Q[i] (affine) for i in [0,n).
// P, Q, R are flat [n × LIMBS_PER_AFFINE].  config: [n].  Test-only.
extern "C" __global__ void batch_affine_add_test(
    const u64limb* __restrict__ pin,
    const u64limb* __restrict__ qin,
    u64limb*       __restrict__ rout,
    const uint*    __restrict__ config,
    const u64limb* __restrict__ field_params
) {
    if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;
    uint n = config[0];
    if (n > BATCH_INV_TEST_MAX) return;

    FieldParams field = load_field(field_params);

    AffinePoint P[BATCH_INV_TEST_MAX];
    AffinePoint Q[BATCH_INV_TEST_MAX];
    AffinePoint R[BATCH_INV_TEST_MAX];
    BigInt denom[BATCH_INV_TEST_MAX];
    BigInt denom_inv[BATCH_INV_TEST_MAX];
    BigInt scratch[BATCH_INV_TEST_MAX];

    for (uint i = 0; i < n; i++) {
        for (uint l = 0; l < NUM_LIMBS; l++) {
            P[i].x.limbs[l] = pin[i * LIMBS_PER_AFFINE + l];
            P[i].y.limbs[l] = pin[i * LIMBS_PER_AFFINE + NUM_LIMBS + l];
            Q[i].x.limbs[l] = qin[i * LIMBS_PER_AFFINE + l];
            Q[i].y.limbs[l] = qin[i * LIMBS_PER_AFFINE + NUM_LIMBS + l];
        }
    }

    batch_affine_add_local(P, Q, R, n, field, denom, denom_inv, scratch);

    for (uint i = 0; i < n; i++) {
        for (uint l = 0; l < NUM_LIMBS; l++) {
            rout[i * LIMBS_PER_AFFINE + l]             = R[i].x.limbs[l];
            rout[i * LIMBS_PER_AFFINE + NUM_LIMBS + l] = R[i].y.limbs[l];
        }
    }
}
