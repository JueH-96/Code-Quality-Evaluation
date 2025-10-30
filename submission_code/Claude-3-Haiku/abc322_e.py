import sys

# Read input
N, K, P = map(int, input().split())
plans = []
for _ in range(N):
    plan = list(map(int, input().split()))
    plans.append(plan)

# Initialize parameter values to 0
params = [0] * K

# Try all possible combinations of development plans
min_cost = float('inf')
for mask in range(1, 2**N):
    cost = 0
    for i in range(N):
        if mask & (1 << i):
            cost += plans[i][0]
            for j in range(K):
                params[j] += plans[i][j+1]
    if all(param >= P for param in params):
        min_cost = min(min_cost, cost)

if min_cost == float('inf'):
    print(-1)
else:
    print(min_cost)