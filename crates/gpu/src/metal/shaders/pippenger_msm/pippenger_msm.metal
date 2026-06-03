#include <metal_stdlib>
using namespace metal;

constant uint NUM_LIMBS = 4;
constant uint COORDS_PER_POINT = 3;
constant uint LIMBS_PER_POINT = NUM_LIMBS * COORDS_PER_POINT;
constant uint MAX_PRIVATE_BUCKETS = 64;

struct BigInt {
    ulong limbs[NUM_LIMBS];
};

struct BigIntWide {
    ulong limbs[NUM_LIMBS * 2];
};

struct FieldParams {
    BigInt modulus;
    ulong inv;
};

struct JacobianPoint {
    BigInt x;
    BigInt y;
    BigInt z;
};

BigInt bigint_zero() {
    BigInt z;
    for (uint i = 0; i < NUM_LIMBS; i++) {
        z.limbs[i] = 0;
    }
    return z;
}

bool bigint_is_zero(BigInt a) {
    for (uint i = 0; i < NUM_LIMBS; i++) {
        if (a.limbs[i] != 0) {
            return false;
        }
    }
    return true;
}

BigInt bigint_add(BigInt a, BigInt b, thread ulong& carry_out) {
    BigInt result;
    ulong carry = 0;
    for (uint i = 0; i < NUM_LIMBS; i++) {
        ulong sum1 = a.limbs[i] + b.limbs[i];
        ulong c1 = sum1 < a.limbs[i] ? 1 : 0;
        ulong sum2 = sum1 + carry;
        ulong c2 = sum2 < sum1 ? 1 : 0;
        result.limbs[i] = sum2;
        carry = c1 + c2;
    }
    carry_out = carry;
    return result;
}

BigInt bigint_sub(BigInt a, BigInt b, thread ulong& borrow_out) {
    BigInt result;
    ulong borrow = 0;
    for (uint i = 0; i < NUM_LIMBS; i++) {
        ulong diff1 = a.limbs[i] - b.limbs[i];
        ulong b1 = a.limbs[i] < b.limbs[i] ? 1 : 0;
        ulong diff2 = diff1 - borrow;
        ulong b2 = diff1 < borrow ? 1 : 0;
        result.limbs[i] = diff2;
        borrow = b1 + b2;
    }
    borrow_out = borrow;
    return result;
}

FieldParams load_field(device const ulong* params) {
    FieldParams field;
    for (uint i = 0; i < NUM_LIMBS; i++) {
        field.modulus.limbs[i] = params[i];
    }
    field.inv = params[NUM_LIMBS];
    return field;
}

BigInt field_add(BigInt a, BigInt b, FieldParams field) {
    ulong carry;
    BigInt sum = bigint_add(a, b, carry);
    ulong borrow;
    BigInt reduced = bigint_sub(sum, field.modulus, borrow);
    return (carry != 0 || borrow == 0) ? reduced : sum;
}

BigInt field_sub(BigInt a, BigInt b, FieldParams field) {
    ulong borrow;
    BigInt diff = bigint_sub(a, b, borrow);
    if (borrow != 0) {
        ulong carry;
        diff = bigint_add(diff, field.modulus, carry);
    }
    return diff;
}

BigInt field_neg(BigInt a, FieldParams field) {
    if (bigint_is_zero(a)) {
        return a;
    }
    ulong borrow;
    return bigint_sub(field.modulus, a, borrow);
}

BigInt field_double(BigInt a, FieldParams field) {
    return field_add(a, a, field);
}

BigInt mont_reduce(BigIntWide t, FieldParams field) {
    BigIntWide tmp = t;

    for (uint i = 0; i < NUM_LIMBS; i++) {
        ulong m = tmp.limbs[i] * field.inv;
        ulong carry = 0;

        for (uint j = 0; j < NUM_LIMBS; j++) {
            ulong lo = m * field.modulus.limbs[j];
            ulong hi = mulhi(m, field.modulus.limbs[j]);

            ulong sum1 = tmp.limbs[i + j] + lo;
            ulong c1 = sum1 < tmp.limbs[i + j] ? 1 : 0;
            ulong sum2 = sum1 + carry;
            ulong c2 = sum2 < sum1 ? 1 : 0;
            tmp.limbs[i + j] = sum2;
            carry = hi + c1 + c2;
        }

        for (uint j = NUM_LIMBS; i + j < NUM_LIMBS * 2; j++) {
            ulong sum = tmp.limbs[i + j] + carry;
            carry = sum < tmp.limbs[i + j] ? 1 : 0;
            tmp.limbs[i + j] = sum;
            if (carry == 0) {
                break;
            }
        }
    }

    BigInt result;
    for (uint i = 0; i < NUM_LIMBS; i++) {
        result.limbs[i] = tmp.limbs[i + NUM_LIMBS];
    }

    ulong borrow;
    BigInt reduced = bigint_sub(result, field.modulus, borrow);
    return borrow == 0 ? reduced : result;
}

BigInt mont_mul(BigInt a, BigInt b, FieldParams field) {
    BigIntWide product;
    for (uint i = 0; i < NUM_LIMBS * 2; i++) {
        product.limbs[i] = 0;
    }

    for (uint i = 0; i < NUM_LIMBS; i++) {
        ulong carry = 0;
        for (uint j = 0; j < NUM_LIMBS; j++) {
            ulong lo = a.limbs[i] * b.limbs[j];
            ulong hi = mulhi(a.limbs[i], b.limbs[j]);
            ulong sum1 = product.limbs[i + j] + lo;
            ulong c1 = sum1 < product.limbs[i + j] ? 1 : 0;
            ulong sum2 = sum1 + carry;
            ulong c2 = sum2 < sum1 ? 1 : 0;
            product.limbs[i + j] = sum2;
            carry = hi + c1 + c2;
        }
        product.limbs[i + NUM_LIMBS] += carry;
    }

    return mont_reduce(product, field);
}

BigInt mont_square(BigInt a, FieldParams field) {
    return mont_mul(a, a, field);
}

JacobianPoint jacobian_identity() {
    JacobianPoint p;
    p.x = bigint_zero();
    p.y = bigint_zero();
    p.z = bigint_zero();
    return p;
}

bool jacobian_is_identity(JacobianPoint p) {
    return bigint_is_zero(p.z);
}

JacobianPoint jacobian_double(JacobianPoint p, FieldParams field) {
    if (jacobian_is_identity(p)) {
        return p;
    }

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
        field
    );
    result.z = field_double(mont_mul(p.y, p.z, field), field);
    return result;
}

JacobianPoint jacobian_add(JacobianPoint p, JacobianPoint q, FieldParams field) {
    if (jacobian_is_identity(p)) {
        return q;
    }
    if (jacobian_is_identity(q)) {
        return p;
    }

    BigInt Z1Z1 = mont_square(p.z, field);
    BigInt Z2Z2 = mont_square(q.z, field);
    BigInt U1 = mont_mul(p.x, Z2Z2, field);
    BigInt U2 = mont_mul(q.x, Z1Z1, field);
    BigInt S1 = mont_mul(mont_mul(p.y, q.z, field), Z2Z2, field);
    BigInt S2 = mont_mul(mont_mul(q.y, p.z, field), Z1Z1, field);
    BigInt H = field_sub(U2, U1, field);

    if (bigint_is_zero(H)) {
        if (bigint_is_zero(field_sub(S2, S1, field))) {
            return jacobian_double(p, field);
        }
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
        field
    );
    result.z = mont_mul(
        field_sub(field_sub(mont_square(field_add(p.z, q.z, field), field), Z1Z1, field), Z2Z2, field),
        H,
        field
    );
    return result;
}

JacobianPoint jacobian_add_mixed(JacobianPoint p, JacobianPoint q, FieldParams field) {
    if (jacobian_is_identity(p)) {
        return q;
    }
    if (jacobian_is_identity(q)) {
        return p;
    }

    BigInt Z1Z1 = mont_square(p.z, field);
    BigInt U1 = p.x;
    BigInt U2 = mont_mul(q.x, Z1Z1, field);
    BigInt S1 = p.y;
    BigInt S2 = mont_mul(mont_mul(q.y, p.z, field), Z1Z1, field);
    BigInt H = field_sub(U2, U1, field);

    if (bigint_is_zero(H)) {
        if (bigint_is_zero(field_sub(S2, S1, field))) {
            return jacobian_double(p, field);
        }
        return jacobian_identity();
    }

    BigInt HH = mont_square(H, field);
    BigInt HHH = mont_mul(H, HH, field);
    BigInt r = field_sub(S2, S1, field);
    BigInt V = mont_mul(U1, HH, field);

    JacobianPoint result;
    result.x = field_sub(field_sub(mont_square(r, field), HHH, field), field_double(V, field), field);
    result.y = field_sub(
        mont_mul(r, field_sub(V, result.x, field), field),
        mont_mul(S1, HHH, field),
        field
    );
    result.z = mont_mul(p.z, H, field);
    return result;
}

JacobianPoint jacobian_neg(JacobianPoint p, FieldParams field) {
    p.y = field_neg(p.y, field);
    return p;
}

JacobianPoint load_point(device const ulong* points, uint point_idx) {
    uint base = point_idx * LIMBS_PER_POINT;
    JacobianPoint p;
    for (uint i = 0; i < NUM_LIMBS; i++) {
        p.x.limbs[i] = points[base + i];
        p.y.limbs[i] = points[base + NUM_LIMBS + i];
        p.z.limbs[i] = points[base + 2 * NUM_LIMBS + i];
    }
    return p;
}

void store_point(device ulong* points, uint point_idx, JacobianPoint p) {
    uint base = point_idx * LIMBS_PER_POINT;
    for (uint i = 0; i < NUM_LIMBS; i++) {
        points[base + i] = p.x.limbs[i];
        points[base + NUM_LIMBS + i] = p.y.limbs[i];
        points[base + 2 * NUM_LIMBS + i] = p.z.limbs[i];
    }
}

kernel void clear_u64_buffer(
    device ulong* buffer [[buffer(0)]],
    device const uint* config [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    uint len = config[0];
    if (gid < len) {
        buffer[gid] = 0;
    }
}

kernel void bucket_accumulation_by_chunk(
    device const int* digits [[buffer(0)]],
    device const ulong* points [[buffer(1)]],
    device ulong* partial_buckets [[buffer(2)]],
    device const uint* config [[buffer(3)]],
    device const ulong* field_params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint num_scalars = config[0];
    uint num_windows = config[1];
    uint num_buckets = config[2];
    uint chunk_size = config[3];
    uint num_chunks = config[4];
    uint total_workers = num_windows * num_chunks;

    if (gid >= total_workers) {
        return;
    }

    uint window_idx = gid / num_chunks;
    uint chunk_idx = gid % num_chunks;
    uint start = chunk_idx * chunk_size;
    uint end = min(start + chunk_size, num_scalars);
    uint partial_base = (window_idx * num_chunks + chunk_idx) * num_buckets;
    FieldParams field = load_field(field_params);

    if (num_buckets <= MAX_PRIVATE_BUCKETS) {
        JacobianPoint local_buckets[MAX_PRIVATE_BUCKETS];
        for (uint bucket_idx = 0; bucket_idx < num_buckets; bucket_idx++) {
            local_buckets[bucket_idx] = jacobian_identity();
        }

        for (uint scalar_idx = start; scalar_idx < end; scalar_idx++) {
            int digit = digits[scalar_idx * num_windows + window_idx];
            if (digit == 0) {
                continue;
            }

            uint bucket_idx = digit > 0 ? uint(digit - 1) : uint(-digit - 1);
            // Input points are affine (z = Montgomery(1) = R mod p).
            // jacobian_add_mixed saves ~5 field multiplications vs full jacobian_add.
            JacobianPoint p = load_point(points, scalar_idx);
            if (digit < 0) {
                p = jacobian_neg(p, field);
            }

            local_buckets[bucket_idx] = jacobian_add_mixed(local_buckets[bucket_idx], p, field);
        }

        for (uint bucket_idx = 0; bucket_idx < num_buckets; bucket_idx++) {
            store_point(partial_buckets, partial_base + bucket_idx, local_buckets[bucket_idx]);
        }
        return;
    }

    for (uint scalar_idx = start; scalar_idx < end; scalar_idx++) {
        int digit = digits[scalar_idx * num_windows + window_idx];
        if (digit == 0) {
            continue;
        }

        uint bucket_idx = digit > 0 ? uint(digit - 1) : uint(-digit - 1);
        JacobianPoint p = load_point(points, scalar_idx);
        if (digit < 0) {
            p = jacobian_neg(p, field);
        }

        uint partial_idx = partial_base + bucket_idx;
        JacobianPoint bucket = load_point(partial_buckets, partial_idx);
        bucket = jacobian_add_mixed(bucket, p, field);
        store_point(partial_buckets, partial_idx, bucket);
    }
}

kernel void bucket_merge(
    device const ulong* partial_buckets [[buffer(0)]],
    device ulong* buckets [[buffer(1)]],
    device const uint* config [[buffer(2)]],
    device const ulong* field_params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint num_windows = config[0];
    uint num_buckets = config[1];
    uint num_chunks = config[2];
    uint total_buckets = num_windows * num_buckets;

    if (gid >= total_buckets) {
        return;
    }

    uint window_idx = gid / num_buckets;
    uint bucket_idx = gid % num_buckets;
    FieldParams field = load_field(field_params);
    JacobianPoint bucket = jacobian_identity();

    for (uint chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
        uint partial_idx = (window_idx * num_chunks + chunk_idx) * num_buckets + bucket_idx;
        JacobianPoint partial = load_point(partial_buckets, partial_idx);
        bucket = jacobian_add(bucket, partial, field);
    }

    store_point(buckets, gid, bucket);
}

kernel void bucket_reduction(
    device const ulong* buckets [[buffer(0)]],
    device ulong* window_sums [[buffer(1)]],
    device const uint* config [[buffer(2)]],
    device const ulong* field_params [[buffer(3)]],
    uint window_idx [[thread_position_in_grid]]
) {
    uint num_windows = config[0];
    uint num_buckets = config[1];

    if (window_idx >= num_windows) {
        return;
    }

    FieldParams field = load_field(field_params);
    uint bucket_base = window_idx * num_buckets;
    JacobianPoint running_sum = jacobian_identity();
    JacobianPoint result = jacobian_identity();

    for (int i = int(num_buckets) - 1; i >= 0; i--) {
        JacobianPoint bucket = load_point(buckets, bucket_base + uint(i));
        running_sum = jacobian_add(running_sum, bucket, field);
        result = jacobian_add(result, running_sum, field);
    }

    store_point(window_sums, window_idx, result);
}
