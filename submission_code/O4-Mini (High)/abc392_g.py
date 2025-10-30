import sys

def main():
    import sys
    input = sys.stdin.readline

    MOD = 998244353
    PRIMITIVE_ROOT = 3

    n = int(input())
    S = list(map(int, input().split()))
    if n < 3:
        print(0)
        return

    maxv = max(S)
    # Convolution size: next power of two ≥ 2*maxv+1
    size = 1
    while size <= 2 * maxv:
        size <<= 1

    # build indicator array
    a = [0] * size
    for x in S:
        a[x] = 1

    # Cooley–Tukey in‐place iterative NTT
    def ntt(f, invert):
        n = len(f)
        # bit‐reversal permutation
        j = 0
        for i in range(1, n):
            bit = n >> 1
            while j & bit:
                j ^= bit
                bit >>= 1
            j |= bit
            if i < j:
                f[i], f[j] = f[j], f[i]
        # layers
        length = 2
        while length <= n:
            wlen = pow(PRIMITIVE_ROOT, (MOD - 1) // length, MOD)
            if invert:
                wlen = pow(wlen, MOD - 2, MOD)
            half = length >> 1
            for i in range(0, n, length):
                w = 1
                for j in range(i, i + half):
                    u = f[j]
                    v = f[j + half] * w % MOD
                    f[j] = (u + v) % MOD
                    f[j + half] = (u - v + MOD) % MOD
                    w = w * wlen % MOD
            length <<= 1
        if invert:
            inv_n = pow(n, MOD - 2, MOD)
            for i in range(n):
                f[i] = f[i] * inv_n % MOD

    # convolution a * a
    b = a[:]         # copy
    ntt(a, False)
    ntt(b, False)
    for i in range(size):
        a[i] = a[i] * b[i] % MOD
    ntt(a, True)

    # a[k] now equals sum_{i+j=k} f[i]*f[j], exactly the ordered‐pairs count
    # we want for each b in S: h = a[2*b], but we need unordered distinct pairs:
    # total += (h - 1) // 2.  Summing gives (sum h - n)//2.
    total = 0
    for x in S:
        total += a[2 * x]

    ans = (total - n) // 2
    print(ans)


if __name__ == "__main__":
    main()