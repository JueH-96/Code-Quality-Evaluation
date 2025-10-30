from itertools import combinations

# Read input
N, K, P = map(int, input().split())
development_plans = [list(map(int, input().split())) for _ in range(N)]

# Initialize the minimum cost to an impossible high value
min_cost = float('inf')
found = False

# Check all possible combinations of development plans
for r in range(1, N + 1):
    for combo in combinations(development_plans, r):
        cost = 0
        parameters = [0] * K
        for plan in combo:
            cost += plan[0]
            for j in range(K):
                parameters[j] += plan[j + 1]
        # Check if the parameters meet or exceed the goal
        if all(p >= P for p in parameters):
            found = True
            min_cost = min(min_cost, cost)

# Output the result
if found:
    print(min_cost)
else:
    print(-1)