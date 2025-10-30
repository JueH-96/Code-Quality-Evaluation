import sys
from collections import defaultdict

# Read input
data = sys.stdin.read().split()
N = int(data[0])
A = list(map(int, data[1:]))

# Compute prefix distinct counts
prefix_dist = [0] * N
freq_pref = defaultdict(int)
dist_pref = 0
for i in range(N):
    if freq_pref[A[i]] == 0:
        dist_pref += 1
    freq_pref[A[i]] += 1  # Increment frequency, but only use check for distinct count
    prefix_dist[i] = dist_pref

# Compute suffix distinct counts
suffix_dist = [0] * N
freq_suf = defaultdict(int)
dist_suf = 0
for i in range(N-1, -1, -1):
    if freq_suf[A[i]] == 0:
        dist_suf += 1
    freq_suf[A[i]] += 1  # Increment frequency, but only use check for distinct count
    suffix_dist[i] = dist_suf

# Compute the maximum sum of distinct counts
max_sum = max(prefix_dist[idx] + suffix_dist[idx + 1] for idx in range(N-1))

# Output the result
print(max_sum)