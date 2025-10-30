import sys
import bisect

# Read all input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
X = int(data[index + 1])
Y = int(data[index + 2])
index += 3
A = [int(data[index + i]) for i in range(N)]
index += N
B = [int(data[index + i]) for i in range(N)]

# Compute cumulative sum for A
sorted_A_desc = sorted(A, reverse=True)
cum_sum_A = [0]
cum = 0
for a in sorted_A_desc:
    cum += a
    cum_sum_A.append(cum)

# Compute cumulative sum for B
sorted_B_desc = sorted(B, reverse=True)
cum_sum_B = [0]
cum = 0
for b in sorted_B_desc:
    cum += b
    cum_sum_B.append(cum)

# Binary search to find the smallest k for A and B
idx_A = bisect.bisect_left(cum_sum_A, X + 1)
idx_B = bisect.bisect_left(cum_sum_B, Y + 1)

# Find the answer
valid_ks = []
if idx_A <= N:
    valid_ks.append(idx_A)
if idx_B <= N:
    valid_ks.append(idx_B)
if valid_ks:
    ans = min(valid_ks)
else:
    ans = N

# Output the answer
print(ans)