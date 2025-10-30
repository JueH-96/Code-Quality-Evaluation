def main():
    import sys
    from functools import lru_cache

    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    weights = [[0] * N for _ in range(N)]
    index = 1

    # Build the full symmetric weight matrix.
    # For i in 0 to N-2, we have weights for vertices i+1 to N.
    for i in range(N - 1):
        for j in range(i + 1, N):
            w = int(data[index])
            index += 1
            weights[i][j] = w
            weights[j][i] = w

    full_mask = (1 << N) - 1

    # Use DP with bitmask representing which vertices have been matched.
    @lru_cache(maxsize=None)
    def dp(mask):
        # If all vertices are used, nothing more to add.
        if mask == full_mask:
            return 0
        # Find the smallest vertex that is not yet used.
        v = 0
        while mask & (1 << v):
            v += 1
        # Option 1: Leave vertex v unmatched.
        best = dp(mask | (1 << v))
        # Option 2: Pair vertex v with any other unmatched vertex.
        for u in range(v + 1, N):
            if not mask & (1 << u):
                best = max(best, weights[v][u] + dp(mask | (1 << v) | (1 << u)))
        return best

    result = dp(0)
    sys.stdout.write(str(result))

if __name__ == '__main__':
    main()