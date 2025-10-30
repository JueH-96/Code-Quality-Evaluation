import sys

def solve():
    N = int(input())
    D = list(map(int, input().split()))
    L = [0] * 2
    C = [0] * 2
    K = [0] * 2
    for i in range(2):
        L[i], C[i], K[i] = map(int, input().split())

    # Calculate the minimum cost to cover a section of length d
    def min_cost(d):
        # Calculate the cost for each type of sensor
        cost = [float('inf')] * 2
        for i in range(2):
            if d <= L[i] * K[i]:
                cost[i] = (d + L[i] - 1) // L[i] * C[i]
        # Return the minimum cost or -1 if it's not possible
        min_cost = min(cost)
        return min_cost if min_cost != float('inf') else -1

    # Calculate the total minimum cost
    total_cost = 0
    for d in D:
        cost = min_cost(d)
        if cost == -1:
            print(-1)
            return
        total_cost += cost
    print(total_cost)

solve()