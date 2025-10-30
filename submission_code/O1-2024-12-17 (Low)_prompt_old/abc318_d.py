def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    
    # Read the upper-triangular weights and fill a symmetric matrix w
    w = [[0]*N for _ in range(N)]
    idx = 1
    for i in range(N-1):
        for j in range(i+1, N):
            weight = int(input_data[idx])
            idx += 1
            w[i][j] = weight
            w[j][i] = weight

    # dp[mask] will hold the maximum sum of chosen edges for the subset of vertices indicated by mask
    dp = [0]*(1<<N)

    # For each subset of vertices
    for mask in range(1<<N):
        # Find the first set bit (smallest vertex in the subset)
        # skip mask=0 where there's no set bit
        # skip also if the subset has fewer than 2 vertices
        if mask == 0:
            continue
        
        # Zero-based: find first set bit
        # (We can do a trick: i = (mask & -mask).bit_length() - 1, but let's do it straightforwardly)
        i = (mask & -mask).bit_length() - 1
        
        # Attempt pairing i with some other j in the subset
        # after we remove i and j from the subset, that subset is new_mask
        # and we can add w[i][j] to dp[new_mask]
        # We'll update dp[mask] to the best found
        # Because dp[mask] should be at least dp[mask] (maybe previously updated),
        # and might improve by pairing i with j.
        
        # Note: we can skip if there's only one vertex in mask (which is i).
        # In that case, there's no j to pick.
        temp_mask = mask ^ (1 << i)  # remove i
        subset = temp_mask
        while subset:
            # j is the lowest set bit of subset
            j = (subset & -subset).bit_length() - 1
            new_mask = mask ^ (1 << i) ^ (1 << j)
            dp[mask] = max(dp[mask], dp[new_mask] + w[i][j])
            subset ^= (1 << j)

    print(dp[(1<<N) - 1])

def main():
    solve()

if __name__ == "__main__":
    main()