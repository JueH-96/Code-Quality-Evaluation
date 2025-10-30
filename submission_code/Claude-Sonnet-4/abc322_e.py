from collections import defaultdict

def solve():
    N, K, P = map(int, input().split())
    plans = []
    
    for _ in range(N):
        line = list(map(int, input().split()))
        cost = line[0]
        params = line[1:]
        plans.append((cost, params))
    
    # DP state: tuple of parameter values -> minimum cost
    # Since we only care about values up to P, we can cap them at P
    dp = defaultdict(lambda: float('inf'))
    dp[tuple([0] * K)] = 0
    
    for cost, params in plans:
        new_dp = dp.copy()
        
        for state, current_cost in dp.items():
            if current_cost == float('inf'):
                continue
                
            # Apply this development plan
            new_state = []
            for i in range(K):
                new_val = min(P, state[i] + params[i])
                new_state.append(new_val)
            new_state = tuple(new_state)
            
            new_cost = current_cost + cost
            if new_cost < new_dp[new_state]:
                new_dp[new_state] = new_cost
        
        dp = new_dp
    
    # Find minimum cost where all parameters are at least P
    min_cost = float('inf')
    for state, cost in dp.items():
        if all(param >= P for param in state):
            min_cost = min(min_cost, cost)
    
    if min_cost == float('inf'):
        print(-1)
    else:
        print(min_cost)

solve()