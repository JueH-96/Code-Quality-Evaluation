# YOUR CODE HERE
import sys
from math import ceil

def min_cost_to_monitor(N, D, L, C, K):
    def can_monitor(x):
        remaining = D.copy()
        for j in range(2):
            for _ in range(K[j]):
                if not remaining:
                    return True
                i = max(range(len(remaining)), key=lambda i: min(remaining[i], L[j]))
                remaining[i] -= L[j]
                if remaining[i] <= 0:
                    remaining.pop(i)
        return not remaining

    def cost(x):
        return sum(C[j] * min(K[j], ceil(x * D_sum / L[j])) for j in range(2))

    D_sum = sum(D)
    left, right = 0, 1
    while not can_monitor(right):
        right *= 2
        if right > 1e9:
            return -1

    while right - left > 1e-9:
        mid = (left + right) / 2
        if can_monitor(mid):
            right = mid
        else:
            left = mid

    return cost(right)

N = int(input())
D = list(map(int, input().split()))
L, C, K = [], [], []
for _ in range(2):
    l, c, k = map(int, input().split())
    L.append(l)
    C.append(c)
    K.append(k)

result = min_cost_to_monitor(N, D, L, C, K)
print(int(result) if result != -1 else -1)