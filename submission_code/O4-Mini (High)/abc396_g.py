import sys
def main():
    import sys
    input = sys.stdin.readline
    H, W = map(int, input().split())
    N = 1 << W

    # freq array: f[mask] = number of rows equal to mask
    f = [0] * N
    for _ in range(H):
        s = input().strip()
        # parse row as binary mask
        m = int(s, 2)
        f[m] += 1

    # cost array: g[mask] = min(popcount(mask), W - popcount(mask))
    # This is the cost contribution of a row that after column-flips
    # yields this mask (before the optional row-flip).
    g = [0] * N
    for i in range(N):
        bc = i.bit_count()
        # either leave row as is, or flip entire row:
        # cost = min(#1s, #0s) = min(bc, W - bc)
        if bc <= W - bc:
            g[i] = bc
        else:
            g[i] = W - bc

    # Fast Walsh–Hadamard Transform for XOR convolution
    def fwt(a):
        n = len(a)
        h = 1
        while h < n:
            for i in range(0, n, h << 1):
                for j in range(i, i + h):
                    u = a[j]
                    v = a[j + h]
                    a[j] = u + v
                    a[j + h] = u - v
            h <<= 1

    def ifwt(a):
        n = len(a)
        h = 1
        # inverse is the same "but divide by 2" at each stage
        while h < n:
            for i in range(0, n, h << 1):
                for j in range(i, i + h):
                    u = a[j]
                    v = a[j + h]
                    # exact divisibility guaranteed by the transform theory
                    a[j]     = (u + v) >> 1
                    a[j + h] = (u - v) >> 1
            h <<= 1

    # Compute XOR convolution h = f (*) g  where
    # h[c] = sum_{mask} f[mask] * g[mask XOR c]
    fwt(f)
    fwt(g)
    # pointwise multiply
    for i in range(N):
        f[i] = f[i] * g[i]
    # inverse transform
    ifwt(f)

    # answer is min over all column-flip patterns c of
    # total cost = f[c]
    ans = min(f)
    print(ans)

if __name__ == "__main__":
    main()