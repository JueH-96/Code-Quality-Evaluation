import sys

# Function to check if a minimum vitamin intake of K is possible
# given precomputed DP tables for each vitamin type.
# dp_v[c] is the maximum vitamin v achievable with at most c calories.
def can_adjusted(K, X, dp1, dp2, dp3):
    # Find the minimum calories required for each vitamin to get at least K units
    # Initialize with a value larger than X
    min_c1 = X + 1
    for c1 in range(X + 1):
        if dp1[c1] >= K:
            min_c1 = c1
            break # Found minimum calories for vitamin 1 >= K

    min_c2 = X + 1
    for c2 in range(X + 1):
        if dp2[c2] >= K:
            min_c2 = c2
            break # Found minimum calories for vitamin 2 >= K

    min_c3 = X + 1
    for c3 in range(X + 1):
        if dp3[c3] >= K:
            min_c3 = c3
            break # Found minimum calories for vitamin 3 >= K

    # If any vitamin cannot reach K units even with X calories (min_c_v remained X+1)
    # or if the sum of minimum calories required exceeds X, then K is not possible.
    if min_c1 > X or min_c2 > X or min_c3 > X or min_c1 + min_c2 + min_c3 > X:
        return False

    # Otherwise, K is possible
    return True

# Read input
N, X = map(int, sys.stdin.readline().split())

foods = []
for _ in range(N):
    V, A, C = map(int, sys.stdin.readline().split())
    foods.append((V, A, C))

# Separate foods by vitamin type
foods_by_type = {1: [], 2: [], 3: []}
for V, A, C in foods:
    foods_by_type[V].append((A, C))

# Compute DP table for each vitamin type
# dp_v[c] will store the maximum vitamin units achievable with at most c calories.
# Initialize DP tables: dp_v[c] = 0 for all c.
dp1 = [0] * (X + 1)
dp2 = [0] * (X + 1)
dp3 = [0] * (X + 1)

# Populate DP tables (0/1 knapsack for "at most c calories")
# Vitamin 1
for A, C in foods_by_type[1]:
    # Iterate calories downwards to ensure each food item is considered once (0/1 knapsack)
    for c in range(X, C - 1, -1):
        dp1[c] = max(dp1[c], dp1[c - C] + A)

# Vitamin 2
for A, C in foods_by_type[2]:
    for c in range(X, C - 1, -1):
        dp2[c] = max(dp2[c], dp2[c - C] + A)

# Vitamin 3
for A, C in foods_by_type[3]:
    for c in range(X, C - 1, -1):
        dp3[c] = max(dp3[c], dp3[c - C] + A)

# Binary search for the maximum K
low = 0
# A safe upper bound for K: The maximum possible value of a single vitamin intake
# cannot exceed the sum of all A_i for that vitamin type, which is bounded by
# N * max(A_i) = 5000 * 2e5 = 1e9. The minimum of three intakes cannot exceed this.
# Use 1e9 + 5 as a sufficiently large upper bound.
high = 1000000000 + 5
ans = 0

while low <= high:
    mid = (low + high) // 2
    if can_adjusted(mid, X, dp1, dp2, dp3):
        ans = mid
        low = mid + 1
    else:
        high = mid - 1

# Print the result
print(ans)