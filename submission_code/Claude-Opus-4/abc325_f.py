# YOUR CODE HERE
def solve():
    N = int(input())
    D = list(map(int, input().split()))
    L1, C1, K1 = map(int, input().split())
    L2, C2, K2 = map(int, input().split())
    
    # For each section, find minimum cost to cover it
    min_costs = []
    total_type1 = 0
    total_type2 = 0
    
    for d in D:
        best_cost = float('inf')
        best_n1 = -1
        best_n2 = -1
        
        # Try all combinations of sensors
        for n1 in range(min(K1 - total_type1 + 1, d // L1 + 2)):
            for n2 in range(min(K2 - total_type2 + 1, d // L2 + 2)):
                if n1 * L1 + n2 * L2 >= d:
                    cost = n1 * C1 + n2 * C2
                    if cost < best_cost:
                        best_cost = cost
                        best_n1 = n1
                        best_n2 = n2
        
        if best_cost == float('inf'):
            print(-1)
            return
        
        min_costs.append((best_cost, best_n1, best_n2))
        total_type1 += best_n1
        total_type2 += best_n2
    
    # Check if we exceeded limits
    if total_type1 > K1 or total_type2 > K2:
        # Need to re-optimize globally
        # Use dynamic programming approach
        
        # dp[i][j][k] = minimum cost to cover first i sections using at most j type-1 and k type-2 sensors
        dp = [[[float('inf')] * (K2 + 1) for _ in range(K1 + 1)] for _ in range(N + 1)]
        dp[0][0][0] = 0
        
        for i in range(N):
            for j in range(K1 + 1):
                for k in range(K2 + 1):
                    if dp[i][j][k] == float('inf'):
                        continue
                    
                    # Try all combinations for section i
                    d = D[i]
                    for n1 in range(min(K1 - j + 1, d // L1 + 2)):
                        for n2 in range(min(K2 - k + 1, d // L2 + 2)):
                            if n1 * L1 + n2 * L2 >= d:
                                new_j = j + n1
                                new_k = k + n2
                                if new_j <= K1 and new_k <= K2:
                                    dp[i + 1][new_j][new_k] = min(dp[i + 1][new_j][new_k], 
                                                                   dp[i][j][k] + n1 * C1 + n2 * C2)
        
        result = float('inf')
        for j in range(K1 + 1):
            for k in range(K2 + 1):
                result = min(result, dp[N][j][k])
        
        if result == float('inf'):
            print(-1)
        else:
            print(result)
    else:
        # Simple greedy works
        print(sum(cost for cost, _, _ in min_costs))

solve()