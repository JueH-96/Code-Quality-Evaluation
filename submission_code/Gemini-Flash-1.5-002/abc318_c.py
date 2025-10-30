# YOUR CODE HERE
n, d, p = map(int, input().split())
f = list(map(int, input().split()))

costs = []
for i in range(n + 1):
    cost = i * p
    fares = sorted(f, reverse=True)
    for j in range(i):
        cost += fares[j]
    costs.append(cost)

min_cost = float('inf')
for i in range(n + 1):
    cost = i * p
    fares = sorted(f, reverse=True)
    remaining_fares = []
    for k in range(n):
        if k < i:
            continue
        remaining_fares.append(f[k])
    cost += sum(remaining_fares)
    min_cost = min(min_cost, cost)

ans = float('inf')
for i in range(n + 1):
    cost = i * (p // d) * d
    if i * (p // d) * d < sum(f):
        temp_f = sorted(f, reverse=True)
        for j in range(i):
            cost += temp_f[j]
        
        
        remaining_fares = []
        for k in range(n):
            if k < i:
                continue
            remaining_fares.append(f[k])
        cost += sum(remaining_fares)
        ans = min(ans, cost)
    else:
        ans = min(ans, cost)

print(min(costs))