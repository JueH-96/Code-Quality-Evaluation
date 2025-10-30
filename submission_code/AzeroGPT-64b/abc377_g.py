import sys
from collections import defaultdict
from bisect import bisect_left

N = int(input())
strings = [input() for _ in range(N)]

# Preprocess
position_indices = defaultdict(list)
for i, string in enumerate(strings):
    position_indices[string[-1]].append(i)

# Problem solving
dp = [0] * N
for k in range(1, N):
    # Start with empty target
    min_cost = len(strings[k])
    # Check previous strings for a possible sequence transformation
    for prev in position_indices[strings[k][-1]]:
        if prev >= k:
            break
        min_cost = min(min_cost, len(strings[k]) - len(strings[prev]) + dp[prev])
    dp[k] = min_cost

sys.stdout.write('
'.join(map(str, dp)))