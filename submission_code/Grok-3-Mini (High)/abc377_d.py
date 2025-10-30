import sys
import bisect

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
M = int(data[index])
index += 1

# Read intervals
intervals = []
for _ in range(N):
    L = int(data[index])
    R = int(data[index + 1])
    intervals.append((L, R))
    index += 2

# Sort intervals by L ascending
sorted_intervals = sorted(intervals, key=lambda x: x[0])

# Extract L_sorted and R_sorted
L_sorted = [x[0] for x in sorted_intervals]
R_sorted = [x[1] for x in sorted_intervals]

# Compute suffix min of R_sorted
N_inter = N
suff_min = [0] * N_inter
suff_min[N_inter - 1] = R_sorted[N_inter - 1]
for j in range(N_inter - 2, -1, -1):
    suff_min[j] = min(R_sorted[j], suff_min[j + 1])

# Compute the sum of valid r for each l
ans = 0
for l in range(1, M + 1):
    # Find the smallest index idx such that L_sorted[idx] >= l
    idx = bisect.bisect_left(L_sorted, l)
    if idx < N_inter:
        A_l = suff_min[idx]
        ub_r = min(M, A_l - 1)
    else:
        ub_r = M
    num_r = max(0, ub_r - l + 1)
    ans += num_r

# Output the answer
print(ans)