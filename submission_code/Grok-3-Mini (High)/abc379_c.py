import sys

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
M = int(data[index + 1])
index += 2
X_list = list(map(int, data[index:index + M]))
index += M
A_list = list(map(int, data[index:index + M]))

# Check sum of A equals N
sum_A = sum(A_list)
if sum_A != N:
    print(-1)
    sys.exit()

# Compute sum of A_i * X_i
sum_AX = sum(x * a for x, a in zip(X_list, A_list))

# Check if minimum X > 1
if min(X_list) > 1:
    print(-1)
    sys.exit()

# Sort the stones by position
stones = list(zip(X_list, A_list))
stones.sort()

# Extract sorted A and compute cumulative sum
A_sorted = [stone[1] for stone in stones]
cum_sum = [0] * M
cum_sum[0] = A_sorted[0]
for i in range(1, M):
    cum_sum[i] = cum_sum[i - 1] + A_sorted[i]

# Start with first stone
P_cur = stones[0][0]
F_cur = cum_sum[0]
val_cur = F_cur - P_cur
if val_cur < 0:
    print(-1)
    sys.exit()

# Check gaps and values for each subsequent stone
for idx in range(1, M):
    P_next = stones[idx][0]
    F_next = cum_sum[idx]
    # Value at the end of the gap
    val_gap_end = F_cur - (P_next - 1)
    if val_gap_end < 0:
        print(-1)
        sys.exit()
    # Value at the next stone position
    val_next = F_next - P_next
    if val_next < 0:
        print(-1)
        sys.exit()
    # Update current values
    val_cur = val_next
    F_cur = F_next
    P_cur = P_next

# If all checks pass, compute and output the cost
sum_final = N * (N + 1) // 2
cost = sum_final - sum_AX
print(cost)