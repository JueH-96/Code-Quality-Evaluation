import sys

# Read N
N = int(sys.stdin.readline())

# Read arrays A, B, and C
A = list(map(int, sys.stdin.readline().split()))
B = list(map(int, sys.stdin.readline().split()))
C = list(map(int, sys.stdin.readline().split()))

total_cost = 0
current_cost_basis = 0

# P_indices will store C_i values for elements A_i=1 that need to become B_i=0.
# These are the "profit" flips (1 -> 0), as they reduce the sum for future operations.
P_indices = [] 

# L_indices will store C_i values for elements A_i=0 that need to become B_i=1.
# These are the "loss" flips (0 -> 1), as they increase the sum for future operations.
L_indices = [] 

# Calculate the initial current_cost_basis (sum(A_k * C_k) for the initial A)
# And populate P_indices and L_indices based on discrepancies between A and B.
for i in range(N):
    if A[i] == 1:
        current_cost_basis += C[i]
    
    if A[i] == 1 and B[i] == 0:
        P_indices.append(C[i])
    elif A[i] == 0 and B[i] == 1:
        L_indices.append(C[i])

# Sort P_indices in descending order. When flipping from 1 to 0, the cost `C_i` is removed
# from the sum. Removing larger `C_i` values first results in a smaller `current_cost_basis`
# for subsequent operations, thus reducing total cost.
P_indices.sort(reverse=True)
    
# Sort L_indices in ascending order. When flipping from 0 to 1, the cost `C_i` is added
# to the sum. Adding smaller `C_i` values first keeps the `current_cost_basis` lower,
# minimizing the cost incurred by these operations.
L_indices.sort()

# Perform all 1->0 flips first. These operations reduce the current_cost_basis.
for c_val in P_indices:
    # A_i flips from 1 to 0, so C_i no longer contributes to the sum.
    # Update current_cost_basis for the state *after* this flip.
    current_cost_basis -= c_val 
    # The cost of this operation is the current state's sum after the flip.
    total_cost += current_cost_basis

# Then perform all 0->1 flips. These operations increase the current_cost_basis.
for c_val in L_indices:
    # A_i flips from 0 to 1, so C_i now contributes to the sum.
    # Update current_cost_basis for the state *after* this flip.
    current_cost_basis += c_val 
    # The cost of this operation is the current state's sum after the flip.
    total_cost += current_cost_basis

# Print the minimum total cost.
print(total_cost)