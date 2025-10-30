def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    
    # Build adjacency matrix W (0-based)
    W = [[0]*N for _ in range(N)]
    idx = 1
    for i in range(N-1):
        for j in range(i+1, N):
            w = int(input_data[idx])
            idx += 1
            W[i][j] = w
            W[j][i] = w
    
    dp_size = 1 << N
    dp = [0] * dp_size
    
    # Bitmask DP for maximum-weight matching
    for mask in range(1, dp_size):
        # Find the first set bit in mask
        # (lowest-indexed vertex included in mask)
        i = 0
        while not (mask & (1 << i)):
            i += 1
        
        # Option 1: Skip vertex i
        m2 = mask ^ (1 << i)
        best_val = dp[m2]
        
        # Option 2: Pair vertex i with some other vertex j in mask
        # and add the corresponding edge weight
        tmp = m2
        while tmp:
            j = (tmp & -tmp).bit_length() - 1
            # j is a set bit in m2, so pair i with j
            m3 = m2 ^ (1 << j)
            val = dp[m3] + W[i][j]
            if val > best_val:
                best_val = val
            tmp ^= (1 << j)
        
        dp[mask] = best_val
    
    # The answer is the value for the full set of vertices (mask = (1<<N) - 1)
    print(dp[(1 << N) - 1])

# Don't forget to call main()
main()