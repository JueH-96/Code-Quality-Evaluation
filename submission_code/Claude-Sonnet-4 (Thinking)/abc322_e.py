N, K, P = map(int, input().split())
plans = []
for _ in range(N):
    line = list(map(int, input().split()))
    cost = line[0]
    increases = line[1:]
    plans.append((cost, increases))

from collections import defaultdict

# dp[state] = minimum cost to achieve state
dp = defaultdict(lambda: float('inf'))
dp[tuple([0] * K)] = 0

for cost, increases in plans:
    new_dp = defaultdict(lambda: float('inf'))
    
    for state, current_cost in dp.items():
        if current_cost == float('inf'):
            continue
        
        # Option 1: Don't execute this plan
        new_dp[state] = min(new_dp[state], current_cost)
        
        # Option 2: Execute this plan
        new_state = tuple(min(P, state[i] + increases[i]) for i in range(K))
        new_dp[new_state] = min(new_dp[new_state], current_cost + cost)
    
    dp = new_dp

# Find minimum cost to achieve goal
min_cost = float('inf')
for state, cost in dp.items():
    if all(p >= P for p in state):
        min_cost = min(min_cost, cost)

if min_cost == float('inf'):
    print(-1)
else:
    print(min_cost)