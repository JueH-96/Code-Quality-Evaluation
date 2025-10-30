# YOUR CODE HERE
import sys
from itertools import combinations

def solve(N, K, P, plans):
    min_cost = float('inf')
    for r in range(1, N + 1):
        for combo in combinations(range(N), r):
            params = [0] * K
            cost = 0
            for idx in combo:
                cost += plans[idx][0]
                for j in range(K):
                    params[j] += plans[idx][j + 1]
            if all(param >= P for param in params):
                min_cost = min(min_cost, cost)
    return min_cost if min_cost != float('inf') else -1

input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
P = int(data[2])
plans = []
for i in range(N):
    plans.append(list(map(int, data[3 + i * (K + 1): 3 + (i + 1) * (K + 1)])))

print(solve(N, K, P, plans))