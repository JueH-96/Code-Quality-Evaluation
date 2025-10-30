import sys

def compute_dp(foods, max_cost):
    dp = [0] * (max_cost + 1)
    for val, cost in foods:
        for c in range(max_cost, cost - 1, -1):
            dp[c] = max(dp[c], dp[c - cost] + val)
    return dp

def find_min_c(dp, M, max_c):
    left = 0
    right = max_c
    result = max_c + 1
    while left <= right:
        mid = (left + right) // 2
        if dp[mid] >= M:
            result = mid
            right = mid - 1
        else:
            left = mid + 1
    return result

def check(M, dp1, dp2, dp3, max_cal):
    c1 = find_min_c(dp1, M, max_cal)
    if c1 > max_cal:
        return False
    c2 = find_min_c(dp2, M, max_cal)
    if c2 > max_cal:
        return False
    c3 = find_min_c(dp3, M, max_cal)
    if c3 > max_cal:
        return False
    total_cost = c1 + c2 + c3
    return total_cost <= max_cal

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
X = int(data[index])
index += 1

foods_v1 = []
foods_v2 = []
foods_v3 = []

for _ in range(N):
    V = int(data[index])
    index += 1
    A = int(data[index])
    index += 1
    C = int(data[index])
    index += 1
    if V == 1:
        foods_v1.append((A, C))
    elif V == 2:
        foods_v2.append((A, C))
    elif V == 3:
        foods_v3.append((A, C))

# Compute maximum possible sums for each vitamin
max_sum1 = sum(a for a, _ in foods_v1)
max_sum2 = sum(a for a, _ in foods_v2)
max_sum3 = sum(a for a, _ in foods_v3)
max_M_possible = min(max_sum1, max_sum2, max_sum3)

# Compute dp for each vitamin
dp_v1 = compute_dp(foods_v1, X)
dp_v2 = compute_dp(foods_v2, X)
dp_v3 = compute_dp(foods_v3, X)

# Binary search for the maximum M
low = 0
high = max_M_possible
while low <= high:
    mid = (low + high) // 2
    if check(mid, dp_v1, dp_v2, dp_v3, X):
        low = mid + 1
    else:
        high = mid - 1

# high is the maximum M where check is true
print(high)