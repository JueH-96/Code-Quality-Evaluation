def solve():
    N = int(input())
    graph = [[0] * (N+1) for _ in range(N+1)]
    
    for i in range(1, N):
        weights = list(map(int, input().split()))
        for j in range(len(weights)):
            graph[i][i+j+1] = weights[j]
            graph[i+j+1][i] = weights[j]
    
    memo = {}
    
    def max_weight_matching(mask):
        if mask in memo:
            return memo[mask]
        
        # Find the first unused vertex
        i = 1
        while i <= N and (mask & (1 << (i-1))):
            i += 1
        
        if i > N:
            return 0
        
        # Option 1: Leave vertex i unmatched
        max_weight = max_weight_matching(mask | (1 << (i-1)))
        
        # Option 2: Match vertex i with another unused vertex
        for j in range(i+1, N+1):
            if not (mask & (1 << (j-1))):
                new_mask = mask | (1 << (i-1)) | (1 << (j-1))
                weight = graph[i][j] + max_weight_matching(new_mask)
                max_weight = max(max_weight, weight)
        
        memo[mask] = max_weight
        return max_weight
    
    return max_weight_matching(0)

print(solve())