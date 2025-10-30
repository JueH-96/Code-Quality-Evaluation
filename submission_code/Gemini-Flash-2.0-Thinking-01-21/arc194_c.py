import sys

# Read N
N = int(sys.stdin.readline())

# Read A, B, C
A = list(map(int, sys.stdin.readline().split()))
B = list(map(int, sys.stdin.readline().split()))
C = list(map(int, sys.stdin.readline().split()))

# Calculate initial sums and collect weights
sum_same1_C = 0 # Sum of C_k for indices k where A_k=1 and B_k=1
sum_diff0_C = 0 # Sum of C_k for indices k where A_k=0 and B_k=1 (Type 01)
sum_diff1_C = 0 # Sum of C_k for indices k where A_k=1 and B_k=0 (Type 10)
weights = []    # Weights for indices in Diff: C_k for Type 10, -C_k for Type 01
m = 0           # Number of indices in Diff

for i in range(N):
    if A[i] == B[i]:
        if A[i] == 1: # A[i]=1, B[i]=1 (Type 11)
            sum_same1_C += C[i]
    else: # A[i] != B[i], must flip this index
        m += 1
        if A[i] == 0: # A[i]=0, B[i]=1 (Type 01)
            sum_diff0_C += C[i]
            weights.append(-C[i])
        else: # A[i]=1, B[i]=0 (Type 10)
            sum_diff1_C += C[i]
            weights.append(C[i])

# Calculate the total cost based on the derived formula
# Total Cost = m * sum(C_k for k in Type 11)
#            + sum(C_k * (p_k - 1) for k in Type 10)
#            + sum(C_k * (m - p_k + 1) for k in Type 01)
# Expand and separate terms dependent on p_k and constant terms:
# Constant part 1: m * sum_same1_C (from Type 11 indices)
# Constant part 2: sum(C_k * (-1) for k in Type 10) + sum(C_k * (m+1) for k in Type 01)
#                = -sum_diff1_C + (m+1) * sum_diff0_C
# Variable part: sum(p_k * C_k for k in Type 10) + sum(p_k * (-C_k) for k in Type 01)
#              = sum(p_k * w_k for k in Diff)
# This variable part is minimized by sorting weights w_k and assigning positions m, m-1, ..., 1.
# The minimum value is sum_{i=0}^{m-1} weights[i] * (m - i)

total_cost = m * sum_same1_C
total_cost += (m + 1) * sum_diff0_C - sum_diff1_C

# Sort weights in ascending order
weights.sort()

# Calculate the minimum sum from the variable part
min_sum_pk_wk = 0
# The weights list has size m. Indices 0 to m-1.
# weights[0] gets position m
# weights[1] gets position m-1
# ...
# weights[m-1] gets position 1
# The sum is weights[0]*m + weights[1]*(m-1) + ... + weights[m-1]*1
for i in range(m):
    min_sum_pk_wk += weights[i] * (m - i)

total_cost += min_sum_pk_wk

# Print the minimum total cost
print(total_cost)