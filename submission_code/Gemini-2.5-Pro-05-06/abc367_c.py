import sys

# Read N and K
N, K = map(int, sys.stdin.readline().split())

# Read R_1, R_2, ..., R_N
R_values = list(map(int, sys.stdin.readline().split()))

results = []  # To store valid sequences
path_build = [] # Stores the sequence being currently constructed

# Precompute suffix sums of R_values for pruning.
# max_suffix_R_sum[i] stores R_values[i] + R_values[i+1] + ... + R_values[N-1].
# It represents the maximum possible sum for elements from index i to N-1.
# Size is N+1 to handle max_suffix_R_sum[N] = 0 (sum of 0 elements).
max_suffix_R_sum = [0] * (N + 1) 
for i in range(N - 1, -1, -1): # Iterate from N-1 down to 0
    max_suffix_R_sum[i] = R_values[i] + max_suffix_R_sum[i+1]

# Recursive function to find sequences
# idx: current index of the element we are trying to determine (0 to N-1)
# current_sum: sum of elements in path_build so far
def find_sequences(idx, current_sum):
    # Base case: if we have constructed a sequence of length N
    if idx == N:
        # All N elements have been chosen. Check sum condition.
        # Since N >= 1 and R_i >= 1, all A_i >= 1, so current_sum will be >= N >= 1.
        # Thus, no need to check if current_sum > 0.
        if current_sum % K == 0:
            results.append(list(path_build)) # Add a copy of the path to results
        return

    # ---- Pruning Logic ----
    # Determine the required sum modulo K for the remaining elements.
    # If current_sum is S_c, and sum of remaining elements is S_rem,
    # we need (S_c + S_rem) % K == 0.
    # So, S_rem % K == (-S_c) % K.
    target_rem_sum_mod_K = -current_sum % K 
    
    # Minimum possible sum for the (N-idx) remaining elements (if all are 1).
    min_val_for_remaining = N - idx 
    # Maximum possible sum for the (N-idx) remaining elements (using R_values from precomputed suffix sums).
    max_val_for_remaining = max_suffix_R_sum[idx]

    # Calculate the smallest possible sum X for the remaining elements,
    # such that X >= min_val_for_remaining AND X % K == target_rem_sum_mod_K.
    
    # Remainder of min_val_for_remaining when divided by K.
    rem_of_min_val = min_val_for_remaining % K
    # Additional value needed to add to min_val_for_remaining to achieve the target_rem_sum_mod_K.
    # This delta must satisfy (rem_of_min_val + delta) % K == target_rem_sum_mod_K.
    # So, delta % K == (target_rem_sum_mod_K - rem_of_min_val + K) % K.
    # The smallest non-negative delta is (target_rem_sum_mod_K - rem_of_min_val + K) % K.
    delta = (target_rem_sum_mod_K - rem_of_min_val + K) % K
    
    achievable_min_sum_with_target_mod = min_val_for_remaining + delta
    
    # If this smallest achievable sum (that satisfies modulo K) is already greater 
    # than the maximum possible sum for remaining elements, then this path is infeasible. Prune.
    if achievable_min_sum_with_target_mod > max_val_for_remaining:
        return 
    # ---- End Pruning Logic ----

    # Recursive step:
    # Iterate through possible values for the element at current_path[idx].
    # The value must be between 1 and R_values[idx] (inclusive).
    for val_choice in range(1, R_values[idx] + 1):
        path_build.append(val_choice) # Choose val_choice for current element
        # Recursively call for the next element
        find_sequences(idx + 1, current_sum + val_choice)
        path_build.pop()  # Backtrack: remove val_choice to try other options

# Initial call to start the recursion:
# Begin with index 0, and current sum 0.
find_sequences(0, 0)

# Print all found sequences.
# They are already in lexicographical order due to the nature of the DFS.
for seq in results:
    # Print elements of the sequence separated by spaces
    print(*(seq))