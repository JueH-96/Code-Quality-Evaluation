# YOUR CODE HERE
N, K, P = map(int, input().split())
plans = []
for i in range(N):
    plan = list(map(int, input().split()))
    plans.append(plan)

min_cost = float('inf')
for i in range(1 << N):
    cost = 0
    params = [0] * K
    selected_plans = []
    for j in range(N):
        if (i >> j) & 1:
            cost += plans[j][0]
            selected_plans.append(plans[j])

    for plan in selected_plans:
        for k in range(K):
            params[k] += plan[k + 1]

    achieved = True
    for param in params:
        if param < P:
            achieved = False
            break

    if achieved:
        min_cost = min(min_cost, cost)

if min_cost == float('inf'):
    print(-1)
else:
    print(min_cost)