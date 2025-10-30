import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
A = list(map(int, data[index:index + N]))
index += N
B = list(map(int, data[index:index + N]))
index += N
C = list(map(int, data[index:index + N]))

# Find differing positions and compute sum for non-differing positions
diff = []
sum_not_D_B_C = 0
for i in range(N):
    if A[i] == B[i]:
        sum_not_D_B_C += B[i] * C[i]
    else:
        diff.append((B[i], C[i]))

M = len(diff)

# Compute sum_D_fixed
sum_D_fixed = 0
for B_k, C_k in diff:
    sum_D_fixed += C_k * (B_k * M + 2 * B_k - 1)

# Compute W list
W = [C_k * (1 - 2 * B_k) for B_k, C_k in diff]

# Sort W in ascending order
W_sorted = sorted(W)

# Compute min sum of w_k * tau_k
min_sum_w_tau = 0
for i in range(M):
    min_sum_w_tau += W_sorted[i] * (M - i)

# Compute total cost
total_cost = (sum_not_D_B_C * M) + sum_D_fixed + min_sum_w_tau

# Output the result
print(total_cost)