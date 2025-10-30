# YOUR CODE HERE
from itertools import combinations
from math import inf

N, K, P = map(int, input().split())
plans = [list(map(int, input().split())) for _ in range(N)]

# Calculate the required increments for each parameter
required = [P] * K
costs = [c for c, *a in plans if all(a[j] >= required[j] for j in range(K))]
plans = [(c, tuple(a)) for c, *a in plans if all(a[j] < required[j] for j in range(K))]

# Sort plans by cost
plans.sort()

# Function to check if a plan can be used
def can_use(plan, used):
    for i, a in enumerate(plan[1]):
        if a > 0 and not used[i]:
            return False
    return True

# Try all combinations of plans
min_cost = inf
for r in range(len(plans) + 1):
    for combo in combinations(plans, r):
        used = [False] * K
        total_cost = 0
        for c, a in combo:
            total_cost += c
            for i, x in enumerate(a):
                if x > 0:
                    used[i] = True
        if all(required[j] <= sum(a[j] for a in combo) for j in range(K)):
            min_cost = min(min_cost, total_cost)

print(min_cost if min_cost != inf else -1)