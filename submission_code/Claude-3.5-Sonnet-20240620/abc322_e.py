# YOUR CODE HERE
from itertools import combinations
import sys

def solve():
    N, K, P = map(int, input().split())
    plans = []
    for _ in range(N):
        plan = list(map(int, input().split()))
        plans.append((plan[0], plan[1:]))

    min_cost = float('inf')
    for r in range(1, N + 1):
        for combo in combinations(plans, r):
            cost = sum(plan[0] for plan in combo)
            params = [sum(plan[1][i] for plan in combo) for i in range(K)]
            if all(param >= P for param in params):
                min_cost = min(min_cost, cost)

    print(min_cost if min_cost != float('inf') else -1)

if __name__ == "__main__":
    solve()