import sys
from itertools import product

input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
index += 1
D = list(map(int, data[index:index + N]))
index += N
L1, C1, K1 = map(int, data[index:index + 3])
index += 3
L2, C2, K2 = map(int, data[index:index + 3])

def min_cost_to_monitor(D, L1, C1, K1, L2, C2, K2):
    min_cost = float('inf')
    for k1 in range(K1 + 1):
        for k2 in range(K2 + 1):
            if k1 * L1 + k2 * L2 >= sum(D):
                cost = k1 * C1 + k2 * C2
                min_cost = min(min_cost, cost)
    return min_cost if min_cost != float('inf') else -1

result = min_cost_to_monitor(D, L1, C1, K1, L2, C2, K2)
print(result)