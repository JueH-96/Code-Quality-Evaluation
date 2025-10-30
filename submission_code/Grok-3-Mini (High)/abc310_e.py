import sys

# Read input from stdin
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
S = data[index]

# Convert S to a list of integers
A = [int(c) for c in S]

# Initialize sum of T(k)
sum_ans = 0

# Compute T for k=1 (1-based index)
T_k = A[0]
sum_ans += T_k

# If N > 1, compute T for k=2 to N
if N > 1:
    prev_T = T_k  # Store T for k=1
    for k in range(2, N + 1):  # k from 2 to N, 1-based
        if A[k - 1] == 0:  # A_k == 0
            T_k_val = k - 1
        else:  # A_k == 1
            T_k_val = (k - 1) - prev_T + 1
        sum_ans += T_k_val
        prev_T = T_k_val  # Update prev_T for next iteration

# Output the result
print(sum_ans)