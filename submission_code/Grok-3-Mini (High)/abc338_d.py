import sys
import math

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
M = int(data[index + 1])
index += 2
X = [int(data[i]) for i in range(index, index + M)]

# Initialize sum of D_cw and difference array
sum_D_cw = 0
diff = [0] * (N + 2)  # Use indices 1 to N+1

# Process each pair of consecutive islands in the tour
for i in range(M - 1):
    A = X[i]
    B = X[i + 1]
    D_cw = (B - A) % N  # Clockwise distance
    sum_D_cw += D_cw
    C_i = N - 2 * D_cw  # Cost coefficient
    S_i = A  # Start edge index
    L_i = D_cw  # Length of the arc
    # Compute end edge index
    temp = (S_i + L_i - 1) % N
    E_i = temp if temp != 0 else N
    # Check if the interval wraps around
    if S_i <= E_i:  # No wrap
        diff[S_i] += C_i
        diff[E_i + 1] -= C_i
    else:  # Wraps around
        # Add for sub-interval [S_i, N]
        diff[S_i] += C_i
        diff[N + 1] -= C_i
        # Add for sub-interval [1, E_i]
        diff[1] += C_i
        diff[E_i + 1] -= C_i

# Compute the prefix sum and find the minimum g(k)
sum_val = 0
min_g = float('inf')
for k in range(1, N + 1):
    sum_val += diff[k]
    g_k = sum_val
    if g_k < min_g:
        min_g = g_k

# Compute the minimum tour length
ans = sum_D_cw + min_g

# Output the answer
print(ans)