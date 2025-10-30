import sys
from bisect import bisect_right

input = sys.stdin.read
data = input().split()

N, Q = map(int, data[0].split())
R = list(map(int, data[1].split()))
queries = list(map(int, data[2:]))

# Precompute the prefix sums of R
prefix_sums = [0]
for r in sorted(R):
    prefix_sums.append(prefix_sums[-1] + r)

# Process each query
results = []
for X in queries:
    # Find the maximum number of sleighs that can be pulled with X reindeer
    idx = bisect_right(prefix_sums, X)
    results.append(idx - 1)

# Output the results
for result in results:
    print(result)