def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    weights = list(map(int, data[1:]))
    
    # Build the full adjacency matrix from the given input
    adj = [[0] * N for _ in range(N)]
    index = 0
    for i in range(N):
        for j in range(i + 1, N):
            adj[i][j] = weights[index]
            adj[j][i] = weights[index]
            index += 1
    
    # Use dynamic programming with bitmasking to solve the problem
    # dp[mask] represents the maximum weight of pairing vertices represented by the mask
    dp = [0] * (1 << N)
    
    # Iterate over all possible masks
    for mask in range(1 << N):
        # Count number of bits set in mask
        count = bin(mask).count('1')
        # We can only pair if we have an even number of vertices selected
        if count % 2 == 1:
            continue
        
        # Try pairing each possible pair (i, j) where i < j
        for i in range(N):
            if not (mask & (1 << i)):  # i must be in the mask
                continue
            for j in range(i + 1, N):
                if not (mask & (1 << j)):  # j must be in the mask
                    continue
                # Form a new mask without i and j
                new_mask = mask & ~(1 << i) & ~(1 << j)
                # Update dp[mask] considering the pair (i, j)
                dp[mask] = max(dp[mask], dp[new_mask] + adj[i][j])
    
    # The answer is the maximum weight for the full mask (all vertices included)
    print(dp[(1 << N) - 1])

if __name__ == "__main__":
    main()