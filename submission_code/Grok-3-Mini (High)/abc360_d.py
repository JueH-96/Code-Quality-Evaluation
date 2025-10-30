import sys
import bisect

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
T = int(data[index])
index += 1
S = data[index]
index += 1
X_values = [int(data[i]) for i in range(index, index + N)]

# Create list of ants with position and direction
ants = [(X_values[i], int(S[i])) for i in range(N)]

# Sort ants by position ascending
sorted_ants = sorted(ants, key=lambda x: x[0])

# Extract sorted positions and directions
pos_sorted = [p for p, d in sorted_ants]
dir_sorted = [d for p, d in sorted_ants]

# Create cumulative sum for direction == 0 (moving left)
cum_sum = [0] * (N + 1)
for i in range(N):
    if dir_sorted[i] == 0:
        cum_sum[i + 1] = cum_sum[i] + 1
    else:
        cum_sum[i + 1] = cum_sum[i]

# Initialize answer
ans = 0

# For each ant in sorted order
for i in range(N):
    if dir_sorted[i] == 1:  # Ant is moving right
        val = pos_sorted[i] + 2 * T
        # Binary search to find upper bound
        idx_bs = bisect.bisect_right(pos_sorted, val)
        R_val = idx_bs - 1  # Upper index with position <= val
        L_idx = i + 1  # Lower index for j > i
        # Check if the range is valid and compute sum of dir==0 in the range
        if 0 <= L_idx and L_idx <= R_val and R_val <= N - 1:
            sum_dir0 = cum_sum[R_val + 1] - cum_sum[L_idx]
            ans += sum_dir0

# Output the answer
print(ans)