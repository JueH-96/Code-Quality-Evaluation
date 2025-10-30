import sys
def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    # Read N, M
    N = int(next(it)); M = int(next(it))
    # Read initial balls A[1..N]
    A = [0] * (N + 2)
    for i in range(1, N+1):
        A[i] = int(next(it))
    # Read B sequence (0-indexed)
    B_seq = [0] * M
    for i in range(M):
        B_seq[i] = int(next(it))
    # Prepare BIT (for dynamic range-add, point-query) and diff array for final prefix
    size = N + 2
    BIT = [0] * (size)
    diff = [0] * (size)
    baseline = 0
    Np = N
    N1 = Np + 1
    bit = BIT
    d = diff
    sz = size
    # Process each operation
    for bi in B_seq:
        b = bi + 1
        # Point query A[b] = A[b] + baseline + BIT.sum(b)
        s = 0
        idx = b
        while idx:
            s += bit[idx]
            idx -= idx & -idx
        k = A[b] + baseline + s
        # Remove k from box b: range_add(b,b, -k)
        v = -k
        d[b]     += v
        d[b+1]   -= v
        # BIT.add(b, v)
        idx = b
        while idx < sz:
            bit[idx] += v
            idx += idx & -idx
        # BIT.add(b+1, -v) == BIT.add(b+1, k)
        vv = k
        idx = b + 1
        while idx < sz:
            bit[idx] += vv
            idx += idx & -idx
        # Full cyclic distribution
        q = k // Np
        r = k - q * Np
        baseline += q
        # Residual distribution of r balls
        if r:
            st = b % Np + 1
            end = st + r - 1
            if end <= Np:
                # [st, end] += 1
                d[st]      += 1
                d[end+1]   -= 1
                # BIT.range_add(st, end, +1)
                idx = st
                while idx < sz:
                    bit[idx] += 1
                    idx += idx & -idx
                idx = end + 1
                while idx < sz:
                    bit[idx] -= 1
                    idx += idx & -idx
            else:
                # [st, Np] += 1
                d[st]    += 1
                d[N1]    -= 1
                idx = st
                while idx < sz:
                    bit[idx] += 1
                    idx += idx & -idx
                idx = N1
                while idx < sz:
                    bit[idx] -= 1
                    idx += idx & -idx
                # [1, end-Np] += 1
                rem = end - Np
                d[1]      += 1
                d[rem+1]  -= 1
                idx = 1
                while idx < sz:
                    bit[idx] += 1
                    idx += idx & -idx
                idx = rem + 1
                while idx < sz:
                    bit[idx] -= 1
                    idx += idx & -idx
    # Build final answer: prefix-sum over diff and combine with A and baseline
    out = []
    ap = out.append
    ps = 0
    for i in range(1, Np+1):
        ps += d[i]
        ap(str(A[i] + baseline + ps))
    sys.stdout.write(" ".join(out))

if __name__ == "__main__":
    main()