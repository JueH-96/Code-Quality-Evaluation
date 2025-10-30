import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
A = [int(data[index + i]) for i in range(N)]

# Compute sum of A
sum_A = sum(A)

# Compute Q and R
Q = sum_A // N
R = sum_A % N

# Compute sum of absolute deviations from Q and count P
sum_abs_Q = 0
P = 0
for num in A:
    sum_abs_Q += abs(num - Q)
    if num >= Q + 1:
        P += 1

# Compute minimum sum of absolute deviations
if P >= R:
    min_sum_abs = sum_abs_Q - R
else:
    min_sum_abs = sum_abs_Q + R - 2 * P

# Compute minimum number of operations
min_ops = min_sum_abs // 2

# Output the result
print(min_ops)