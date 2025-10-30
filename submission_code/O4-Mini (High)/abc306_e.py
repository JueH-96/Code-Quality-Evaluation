import sys
def main():
    import sys
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    N = int(next(it)); K = int(next(it)); Q = int(next(it))
    # Read queries and collect values for compression
    Xs = [0]*Q
    Ys = [0]*Q
    vals = {0}
    for i in range(Q):
        x = int(next(it)); y = int(next(it))
        Xs[i] = x; Ys[i] = y
        vals.add(y)
    # Coordinate compress all values
    all_vals = sorted(vals)
    M = len(all_vals)
    comp = {v:i+1 for i,v in enumerate(all_vals)}  # 1-based indices
    # BIT arrays of size M
    bit_n = M
    bit_count = [0] * (bit_n+1)
    bit_sum   = [0] * (bit_n+1)
    # BIT add and sum routines
    def bit_add_count(i, v):
        while i <= bit_n:
            bit_count[i] += v
            i += i & -i
    def bit_add_sum(i, v):
        while i <= bit_n:
            bit_sum[i] += v
            i += i & -i
    def bit_prefix_count(i):
        s = 0
        while i>0:
            s += bit_count[i]
            i -= i & -i
        return s
    def bit_prefix_sum(i):
        s = 0
        while i>0:
            s += bit_sum[i]
            i -= i & -i
        return s
    # Precompute mask for find_kth
    # find_kth: smallest idx with prefix-count >= k
    max_bit = 1 << (bit_n.bit_length() - 1)
    def bit_find(k):
        idx = 0
        b = max_bit
        while b:
            t = idx + b
            if t <= bit_n and bit_count[t] < k:
                idx = t
                k -= bit_count[t]
            b >>= 1
        return idx + 1

    # Initialize: all A[i]=0
    # count of zero = N
    zidx = comp[0]
    bit_add_count(zidx, N)
    # BIT sum for zeros adds nothing
    A = [0] * (N+1)
    total = 0
    C = N - K  # number of smallest to exclude
    out = []

    if K == N:
        # All elements always in the top-K
        for i in range(Q):
            x = Xs[i]; y = Ys[i]
            old = A[x]
            A[x] = y
            total += (y - old)
            out.append(str(total))
    else:
        # General case
        for i in range(Q):
            x = Xs[i]; y = Ys[i]
            old = A[x]
            A[x] = y
            # remove old
            oi = comp[old]
            bit_add_count(oi, -1)
            bit_add_sum(oi, -old)
            # add new
            ni = comp[y]
            bit_add_count(ni, 1)
            bit_add_sum(ni, y)
            total += (y - old)
            # sum of smallest C items
            if C > 0:
                th = bit_find(C)
                cnt_before = bit_prefix_count(th-1)
                sum_before = bit_prefix_sum(th-1)
                take_here = C - cnt_before
                small_sum = sum_before + take_here * all_vals[th-1]
            else:
                small_sum = 0
            # top-K sum = total - sum_smallest(C)
            out.append(str(total - small_sum))

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()