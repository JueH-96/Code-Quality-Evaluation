def main():
    from math import log2
    from functools import lru_cache

    *A, Y = list(map(int, input().split()))
    N, A, X = A
    logN = int(log2(N)) + 3

    @lru_cache(None)
    def dp(n):
        if n < A:
            return 0

        P = [(i, 1 / 6) for i in range(1, 7)] + [(A, X / Y)]
        e = sum(dp(n // d) * p + Y * p for d, p in P)
        return min(e, X + dp(n // A))

    for i in range(logN):
        for j in range(1, 7):
            dp(N // j)

    print("%.18f" % dp(N))

main()