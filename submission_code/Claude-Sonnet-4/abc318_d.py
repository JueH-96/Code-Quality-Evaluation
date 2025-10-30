# YOUR CODE HERE
def solve():
    n = int(input())
    
    # Read the distance matrix
    D = {}
    for i in range(1, n):
        row = list(map(int, input().split()))
        for j in range(len(row)):
            D[(i, i + j + 1)] = row[j]
    
    # DP with bitmask
    # dp[mask] = maximum weight when vertices in mask are already used
    dp = {}
    dp[0] = 0
    
    for mask in range(1 << n):
        if mask not in dp:
            continue
            
        current_weight = dp[mask]
        
        # Find first unused vertex
        first_unused = -1
        for i in range(n):
            if not (mask & (1 << i)):
                first_unused = i
                break
        
        if first_unused == -1:
            continue
            
        # Try pairing first_unused with all other unused vertices
        for j in range(first_unused + 1, n):
            if not (mask & (1 << j)):
                # Can pair vertex first_unused with vertex j
                new_mask = mask | (1 << first_unused) | (1 << j)
                edge_weight = D.get((first_unused + 1, j + 1), 0)
                new_weight = current_weight + edge_weight
                
                if new_mask not in dp or dp[new_mask] < new_weight:
                    dp[new_mask] = new_weight
    
    # Find maximum weight among all possible final states
    max_weight = 0
    for mask in dp:
        max_weight = max(max_weight, dp[mask])
    
    print(max_weight)

solve()