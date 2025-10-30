def main():
    import sys
    from functools import lru_cache

    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    
    # Build a weighted complete graph represented as an N x N matrix.
    # Vertices are 0-indexed.
    D = [[0]*N for _ in range(N)]
    # The input order: first line has D₁,₂ D₁,₃ ... D₁,N,
    # then next line has D₂,₃ ... D₂,N, and so on.
    for i in range(N-1):
        for j in range(i+1, N):
            w = int(next(it))
            D[i][j] = w
            D[j][i] = w

    # We want to choose edges (i,j) such that no vertex is repeated.
    # In other words, we are selecting a matching with maximum total weight.
    # We use a DP with bitmask where each bit represents if the vertex is still available.
    # Let dp(mask) be the maximum sum we can achieve using vertices in 'mask'.
    @lru_cache(maxsize=None)
    def dp(mask):
        if mask == 0:
            return 0
        
        # Find the smallest vertex in mask.
        # (mask & -mask) isolates the lowest set bit.
        i = (mask & -mask).bit_length() - 1
        
        # Option 1: skip pairing vertex i.
        best = dp(mask & ~(1 << i))
        rest = mask & ~(1 << i)
        
        # Option 2: pair vertex i with any other vertex j.
        m = rest
        while m:
            # Get the lowest set bit in m (which represents vertex j).
            j = (m & -m).bit_length() - 1
            candidate = dp(rest & ~(1 << j)) + D[i][j]
            if candidate > best:
                best = candidate
            m &= (m - 1)  # remove the lowest set bit
        return best

    full_mask = (1 << N) - 1
    ans = dp(full_mask)
    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()