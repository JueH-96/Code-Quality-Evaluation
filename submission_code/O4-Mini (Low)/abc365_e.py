def main():
    import sys
    input = sys.stdin.readline

    N = int(input())
    A = list(map(int, input().split()))

    # Build prefix XOR array P of length N+1, P[0]=0, P[i]=A[0]^...^A[i-1]
    P = [0] * (N + 1)
    for i in range(1, N + 1):
        P[i] = P[i - 1] ^ A[i - 1]

    # Count, for each bit k, how many A's have bit k = 1
    max_bit = 30  # since A[i] <= 1e8 < 2^27, 30 is safe
    cntA = [0] * (max_bit + 1)
    for x in A:
        b = 0
        while x:
            if x & 1:
                cntA[b] += 1
            x >>= 1
            b += 1

    # Count, for each bit k, how many prefixes in P have bit k = 1
    cntP = [0] * (max_bit + 1)
    for x in P:
        for k in range(max_bit + 1):
            if (x >> k) & 1:
                cntP[k] += 1

    total = 0
    # Total number of prefixes
    M = N + 1
    for k in range(max_bit + 1):
        c1 = cntP[k]
        c0 = M - c1
        # total differing pairs in prefixes u<v is c0 * c1
        # but we must exclude the immediate neighbors u, u+1:
        # those contribute exactly cntA[k] bad pairs
        good_pairs = c0 * c1 - cntA[k]
        total += good_pairs * (1 << k)

    print(total)


if __name__ == "__main__":
    main()