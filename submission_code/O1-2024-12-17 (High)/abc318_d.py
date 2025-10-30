def main():
    import sys
    input = sys.stdin.readline
    
    N = int(input().strip())
    # Read input weights into a 2D matrix (0-based index)
    W = [[0]*N for _ in range(N)]
    for i in range(N-1):
        row = list(map(int, input().split()))
        for k, val in enumerate(row):
            j = i + 1 + k
            W[i][j] = val
            W[j][i] = val
    
    # dp[mask] will hold the maximum sum of chosen edges (matching) 
    # using only the vertices in 'mask'.
    dp = [0]*(1 << N)
    
    # Iterate over all subsets of vertices
    for mask in range(1, 1 << N):
        # Pick the lowest-set bit (vertex) in mask (call it i)
        i = (mask & -mask).bit_length() - 1
        # Subset mask without vertex i
        mask_without_i = mask ^ (1 << i)
        
        # Option 1: Skip i (i not matched)
        best = dp[mask_without_i]
        
        # Option 2: Pair i with some other j in mask
        x = mask_without_i
        while x > 0:
            j = (x & -x).bit_length() - 1
            candidate = dp[mask_without_i ^ (1 << j)] + W[i][j]
            if candidate > best:
                best = candidate
            x &= (x - 1)
        
        dp[mask] = best
    
    # The answer is the best we can do using all vertices 
    # (we may skip some in the matching if it leads to a higher weight).
    print(dp[(1 << N) - 1])

# Do not forget to call main()
if __name__ == "__main__":
    main()