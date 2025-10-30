import sys

# ---------- Boolean-matrix utilities ---------- #
def mat_mul(A, B, sz):
    """Boolean product C = A * B under (OR,AND) semiring.
       Matrices are given as list[sz] of int bitmasks (row wise)."""
    C = [0] * sz
    for r in range(sz):
        row = A[r]
        res = 0
        while row:
            lsb = row & -row          # lowest set bit
            c = (lsb.bit_length() - 1)
            res |= B[c]
            row -= lsb
        C[r] = res
    return C


def mat_vec_mul(M, v, sz):
    """y = M * v  (boolean) .  Vector stored as bitmask."""
    res = 0
    for r in range(sz):
        if M[r] & v:
            res |= 1 << r
    return res


def build_base_matrices(A, Bmax):
    """Return 1-step matrices for good / bad squares."""
    sz = Bmax
    good = [0] * sz
    bad  = [0] * sz

    # row 0 : new bit
    mask = 0
    for i in range(A - 1, Bmax):
        mask |= 1 << i
    good[0] = mask
    bad[0] = 0                       # cannot land on bad square

    # shift rows
    for r in range(1, sz):
        bit = 1 << (r - 1)
        good[r] = bit
        bad[r]  = bit
    return good, bad


def build_powers(base, max_pow, sz):
    """pre-compute base^(2^k) for k up to max_pow-1"""
    powers = [base]
    for _ in range(1, max_pow):
        powers.append(mat_mul(powers[-1], powers[-1], sz))
    return powers


def apply_powers(powers, v, length, sz):
    """apply matrix^(length) to vector v"""
    k = 0
    while length:
        if length & 1:
            v = mat_vec_mul(powers[k], v, sz)
        length >>= 1
        k += 1
    return v


# ---------- Main solving routine ---------- #
def main():
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    it = iter(input_data)

    N  = int(next(it))
    M  = int(next(it))
    A  = int(next(it))
    Bm = int(next(it))               # maximum step (called B in statement)

    bad_intervals = []
    for _ in range(M):
        L = int(next(it))
        R = int(next(it))
        bad_intervals.append((L, R))

    # Size of state vector = Bm (<=20)
    SZ = Bm

    # Build 1-step matrices
    M_good, M_bad = build_base_matrices(A, Bm)

    # Maximum exponent power needed (>= log2(N))
    MAX_POW = N.bit_length()
    good_powers = build_powers(M_good, MAX_POW, SZ)
    bad_powers  = build_powers(M_bad,  MAX_POW, SZ)

    # Build run list: (is_good, length)
    runs = []
    prev = 1
    for L, R in bad_intervals:
        # good segment before this bad interval
        good_len = L - prev
        if good_len > 0:
            runs.append((True, good_len))
        # the bad interval itself
        runs.append((False, R - L + 1))
        prev = R + 1
    # final good segment up to N
    if prev <= N:
        runs.append((True, N - prev + 1))

    # Initial vector : only square 1 reachable
    vec = 1  # bit 0 set
    first = True

    for is_good, length in runs:
        # skip square 1 (already processed) in the very first good segment
        if first:
            length -= 1
            first = False
            if length == 0:
                continue
        if length == 0:
            continue
        powers = good_powers if is_good else bad_powers
        vec = apply_powers(powers, vec, length, SZ)

    # After processing N squares, bit 0 corresponds to square N
    print("Yes" if (vec & 1) else "No")


if __name__ == "__main__":
    main()