N, M = map(int, input().split())
A = list(map(int, input().split()))

# Compute prefix sums
prefix = [0] * (N + 1)
for i in range(N):
    prefix[i + 1] = prefix[i] + A[i]

total = prefix[N]

# Compute remainders
remainder = [prefix[i] % M for i in range(N)]

# Group positions by remainder
from collections import defaultdict
positions = defaultdict(list)
for i in range(N):
    positions[remainder[i]].append(i + 1)  # 1-indexed positions

count = 0

# Case 1: t > s and remainder[s-1] = remainder[t-1]
for r in positions:
    n = len(positions[r])
    count += n * (n - 1) // 2

# Case 2: t < s and remainder[t-1] = (remainder[s-1] - total) % M
import bisect
if total % M == 0:
    # r1 == r2 for all r1, so we want pairs (s, t) with t < s and remainder[s-1] = remainder[t-1]
    # This is the same as case 1 but with reversed order
    for r in positions:
        n = len(positions[r])
        count += n * (n - 1) // 2
else:
    # r1 != r2 for all r1
    for r1 in positions:
        r2 = (r1 - total) % M
        if r2 in positions:
            # For each s in positions[r1], count t in positions[r2] with t < s
            for s in positions[r1]:
                # Use binary search to find how many t in positions[r2] satisfy t < s
                idx = bisect.bisect_left(positions[r2], s)
                count += idx

print(count)