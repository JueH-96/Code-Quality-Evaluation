N, K, P = map(int, input().split())
plans = []
for _ in range(N):
    line = list(map(int, input().split()))
    plans.append((line[0], line[1:]))

INF = float('inf')
dp = {tuple([0] * K): 0}

for i in range(N):
    cost, gains = plans[i]
    new_dp = dp.copy()
    
    for state in dp:
        new_state = list(state)
        for j in range(K):
            new_state[j] = min(P, state[j] + gains[j])
        new_state = tuple(new_state)
        
        new_cost = dp[state] + cost
        if new_state not in new_dp:
            new_dp[new_state] = new_cost
        else:
            new_dp[new_state] = min(new_dp[new_state], new_cost)
            
    dp = new_dp

min_cost = INF
target = tuple([P] * K)

for state in dp:
    if all(state[i] >= P for i in range(K)):
        min_cost = min(min_cost, dp[state])

print(-1 if min_cost == INF else min_cost)