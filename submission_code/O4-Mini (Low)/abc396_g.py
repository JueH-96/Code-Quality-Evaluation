import sys
import threading
def main():
    import sys
    data = sys.stdin.read().split()
    H = int(data[0]); W = int(data[1])
    rows = data[2:]
    M = 1 << W
    # freq[mask]: number of rows equal to mask
    freq = [0] * M
    for s in rows:
        m = int(s, 2)
        freq[m] += 1
    # kernel K[mask] = min(popcount(mask), W-popcount(mask))
    # cost per row for XOR difference = gamming weight
    K = [0] * M
    # precompute popcounts
    pop = [0] * M
    for i in range(1, M):
        pop[i] = pop[i >> 1] + (i & 1)
    for mask in range(M):
        k = pop[mask]
        K[mask] = k if k <= W - k else W - k

    # XOR-convolution via FWHT
    def fwht(a, invert):
        n = len(a)
        h = 1
        while h < n:
            for i in range(0, n, h * 2):
                for j in range(i, i + h):
                    x = a[j]
                    y = a[j + h]
                    a[j] = x + y
                    a[j + h] = x - y
            h <<= 1
        if invert:
            # divide all by n
            # since n is power of two
            for i in range(n):
                a[i] //= n

    # prepare copies
    A = freq[:]  # will be transformed
    B = K[:]     # will be transformed
    fwht(A, False)
    fwht(B, False)
    # pointwise multiply
    C = [A[i] * B[i] for i in range(M)]
    # inverse transform to get convolution
    fwht(C, True)
    # Now C[c] = sum_{m} freq[m] * min(popcount(m^c), W-popcount(m^c))
    # We need minimal such sum over c
    ans = min(C)
    print(ans)

if __name__ == "__main__":
    main()