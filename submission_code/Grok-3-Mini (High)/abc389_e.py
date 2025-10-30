import sys
from collections import Counter

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
M = int(data[index + 1])
index += 2
P_values = list(map(int, data[index:index + N]))

# Group P_values by their values using Counter
P_counter = Counter(P_values)

def compute_F(W):
    # Compute the number of unit slots with marginal cost <= W
    return sum(cnt * (((W // p) + 1) // 2) for p, cnt in P_counter.items())

def compute_G(W):
    # Compute the sum of marginal costs <= W
    return sum(cnt * (((((W // p) + 1) // 2) ** 2) * p) for p, cnt in P_counter.items())

def check(S, M):
    # Check if the minimum cost to buy S units is <= M
    if S == 0:
        return True
    # Binary search to find the minimum V such that F(V) >= S
    V_low = 0
    V_high = 4000000000000000000  # 4e18
    while V_low < V_high:
        V_mid = (V_low + V_high) // 2
        F_mid = compute_F(V_mid)
        if F_mid >= S:
            V_high = V_mid
        else:
            V_low = V_mid + 1
    V_min = V_low
    W = V_min - 1
    F_W = compute_F(W)
    need = S - F_W
    G_W = compute_G(W)
    sum_cost = G_W + need * V_min
    return sum_cost <= M

# Binary search to find the maximum S such that min cost <= M
low_s = 0
high_s = M
result = -1
while low_s <= high_s:
    mid_s = (low_s + high_s) // 2
    if check(mid_s, M):
        result = mid_s
        low_s = mid_s + 1
    else:
        high_s = mid_s - 1

# Output the result
print(result)