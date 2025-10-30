import sys
import bisect

# Read all input
data = sys.stdin.read().split()
index = 0

# Read N
N = int(data[index])
index += 1

# Read array A
A = [int(data[i]) for i in range(index, index + N)]
index += N

# Read Q
Q = int(data[index])
index += 1

# Calculate number of sleep intervals
M = (N - 1) // 2

# Create start and end sleep times
start_sleep = [0] * M
end_sleep = [0] * M
for i in range(M):
    start_sleep[i] = A[2 * i + 1]
    end_sleep[i] = A[2 * i + 2]

# Compute durations
dur = [end_sleep[i] - start_sleep[i] for i in range(M)]

# Compute prefix sum of durations
prefix_sum_dur = [0]
for d in dur:
    prefix_sum_dur.append(prefix_sum_dur[-1] + d)

# Function to compute cumulative sleep up to time t
def F(t):
    pos = bisect.bisect_right(end_sleep, t)
    num_full = pos  # Number of intervals fully included
    sum_full = prefix_sum_dur[num_full]
    if num_full < M and start_sleep[num_full] <= t:
        sum_partial = t - start_sleep[num_full]
    else:
        sum_partial = 0
    return sum_full + sum_partial

# Process each query
for _ in range(Q):
    L = int(data[index])
    R = int(data[index + 1])
    index += 2
    ans = F(R) - F(L)
    print(ans)