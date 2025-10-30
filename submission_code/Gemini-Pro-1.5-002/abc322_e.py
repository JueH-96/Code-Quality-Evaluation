# YOUR CODE HERE
def solve():
    n, k, p = map(int, input().split())
    costs = []
    benefits = []
    for _ in range(n):
        line = list(map(int, input().split()))
        costs.append(line[0])
        benefits.append(line[1:])

    min_cost = float('inf')
    
    for i in range(1 << n):
        cost = 0
        params = [0] * k
        
        for j in range(n):
            if (i >> j) & 1:
                cost += costs[j]
                for l in range(k):
                    params[l] += benefits[j][l]
        
        achieved = True
        for param in params:
            if param < p:
                achieved = False
                break
        
        if achieved:
            min_cost = min(min_cost, cost)

    if min_cost == float('inf'):
        print(-1)
    else:
        print(min_cost)

solve()