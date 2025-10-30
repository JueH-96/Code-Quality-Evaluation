def solve():
    n, k, p = map(int, input().split())
    plans = []
    for _ in range(n):
        plans.append(list(map(int, input().split())))

    min_cost = float('inf')
    for i in range(1 << n):
        cost = 0
        params = [0] * k
        for j in range(n):
            if (i >> j) & 1:
                cost += plans[j][0]
                for l in range(k):
                    params[l] += plans[j][l + 1]

        if all(param >= p for param in params):
            min_cost = min(min_cost, cost)

    if min_cost == float('inf'):
        print(-1)
    else:
        print(min_cost)

solve()