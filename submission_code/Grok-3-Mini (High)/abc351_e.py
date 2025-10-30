import sys
from collections import defaultdict

# Read all input from stdin
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1

# Group dictionaries for U and V values
u_groups = defaultdict(list)
v_groups = defaultdict(list)

# Read each point and compute S, D, s_mod, d_mod, U, V
for _ in range(N):
    X = int(data[index])
    index += 1
    Y = int(data[index])
    index += 1
    S = X + Y
    D = X - Y
    s_mod = S % 2
    d_mod = D % 2
    U = S // 2  # Floor division
    V = D // 2  # Floor division
    key = (s_mod, d_mod)
    u_groups[key].append(U)
    v_groups[key].append(V)

# Initialize total sum
total_sum = 0

# Iterate over each group
for key in u_groups:
    U_list = u_groups[key]
    V_list = v_groups[key]
    M = len(U_list)  # Number of points in the group
    if M <= 1:
        continue  # No pairs if 0 or 1 point
    
    # Sort the U and V lists
    sorted_U = sorted(U_list)
    sorted_V = sorted(V_list)
    
    # Compute sum of absolute differences for U
    sum_U_diff = 0
    for i in range(M):
        coeff = 2 * (i + 1) - M - 1
        sum_U_diff += coeff * sorted_U[i]
    
    # Compute sum of absolute differences for V
    sum_V_diff = 0
    for i in range(M):
        coeff = 2 * (i + 1) - M - 1
        sum_V_diff += coeff * sorted_V[i]
    
    # Add the group distance sum to total sum
    group_dist_sum = sum_U_diff + sum_V_diff
    total_sum += group_dist_sum

# Output the total sum
print(total_sum)