import sys

def main() -> None:
    it = iter(sys.stdin.read().split())
    N = int(next(it))
    # read the upper‐triangular part and build a symmetric matrix
    w = [[0] * N for _ in range(N)]
    for i in range(N - 1):
        for j in range(i + 1, N):
            val = int(next(it))
            w[i][j] = w[j][i] = val

    size = 1 << N
    NEG = -10 ** 18
    dp = [NEG] * size
    dp[0] = 0                     # empty matching

    for mask in range(size):
        # only masks with even number of set bits can represent a matching
        if mask and (mask.bit_count() & 1):
            continue
        if dp[mask] == NEG:
            continue

        # find the first unused vertex inside `mask`’s complement
        # to extend the current matching
        # However, the classical iterative DP fills dp[mask] itself
        # by pairing vertices already in `mask`.
        # Here we fill dp[mask] when all vertices in `mask` are matched.
        # So we need to build bigger masks from smaller ones:
        first = None
        for v in range(N):
            if not (mask & (1 << v)):
                first = v
                break
        if first is None:            # all vertices are already considered
            continue

        # try pairing `first` with any other unused vertex `u`
        for u in range(first + 1, N):
            if mask & (1 << u):
                continue
            nmask = mask | (1 << first) | (1 << u)
            cand = dp[mask] + w[first][u]
            if cand > dp[nmask]:
                dp[nmask] = cand

    print(max(dp))

if __name__ == "__main__":
    main()