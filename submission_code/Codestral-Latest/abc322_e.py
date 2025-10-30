import sys
from itertools import product

input = sys.stdin.read
data = input().split()

n = int(data[0])
k = int(data[1])
p = int(data[2])

costs = []
increments = []

for i in range(n):
    costs.append(int(data[3 + i * (k + 1)]))
    increments.append([int(data[3 + i * (k + 1) + j]) for j in range(1, k + 1)])

min_cost = float('inf')
found = False

for plan in product([0, 1], repeat=n):
    if sum(plan) == 0:
        continue
    total_cost = sum(costs[i] for i in range(n) if plan[i])
    total_increments = [sum(increments[i][j] for i in range(n) if plan[i]) for j in range(k)]
    if all(total >= p for total in total_increments):
        min_cost = min(min_cost, total_cost)
        found = True

if found:
    print(min_cost)
else:
    print(-1)