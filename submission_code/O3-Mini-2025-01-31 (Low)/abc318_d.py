def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    it = iter(input_data)
    n = int(next(it))
    
    # Create the full symmetric matrix for the weighted complete graph.
    D = [[0] * n for _ in range(n)]
    for i in range(n - 1):
        for j in range(i + 1, n):
            weight = int(next(it))
            D[i][j] = weight
            D[j][i] = weight

    # We'll use a bitmask dynamic programming approach.
    # dp[mask] is the maximum total weight achievable where each bit in 'mask'
    # indicates that the corresponding vertex has been used (i.e., is an endpoint of an edge).
    dp = [0] * (1 << n)
    
    # Process all mask states.
    for mask in range(1 << n):
        # Find the first vertex that is still free (not used in mask).
        v = None
        for i in range(n):
            if not (mask >> i) & 1:
                v = i
                break
        # If no free vertex exists, skip.
        if v is None:
            continue
        
        # Mark vertex v as used.
        new_mask_base = mask | (1 << v)
        
        # Try pairing vertex v with any other free vertex j.
        for j in range(v + 1, n):
            if not (mask >> j) & 1:
                new_mask = new_mask_base | (1 << j)
                dp[new_mask] = max(dp[new_mask], dp[mask] + D[v][j])
    
    # Since we are allowed to select any set of edges (even if not all vertices are used),
    # the answer will be the maximum value in dp over all masks.
    answer = max(dp)
    sys.stdout.write(str(answer))
    
if __name__ == '__main__':
    main()