import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Step 1: Precompute pos[k] = [first_idx, second_idx] for each k
    # pos is a list of lists. pos[k] stores the indices for value k.
    # Initialize with empty lists.
    pos = [[] for _ in range(N + 1)]
    for i in range(2 * N):
        pos[A[i]].append(i)

    # Step 2: Identify "bad" values (where occurrences are already adjacent)
    # These values cannot be 'a' or 'b' in a valid pair.
    bad_vals = set()
    for k in range(1, N + 1):
        if pos[k][1] - pos[k][0] == 1:
            bad_vals.add(k)

    # Step 3: Create mappings for L[k] to k and R[k] to k for efficient lookup
    # Only store mappings for non-bad values.
    first_pos_to_val = {}  # Maps pos[k][0] (first index) to value k
    second_pos_to_val = {} # Maps pos[k][1] (second index) to value k
    
    for k in range(1, N + 1):
        if k not in bad_vals:
            first_pos_to_val[pos[k][0]] = k
            second_pos_to_val[pos[k][1]] = k

    # Step 4: Count valid pairs (a, b)
    count = 0
    
    # Iterate through potential 'a' values (k_val)
    for k_val in range(1, N + 1):
        # Skip if k_val is a bad value (violates Condition 1)
        if k_val in bad_vals:
            continue

        L_k = pos[k_val][0]  # First occurrence of k_val
        R_k = pos[k_val][1]  # Second occurrence of k_val

        # We are looking for a j_val such that:
        # L[j_val] = L[k_val] + 1  (immediately after k_val's first occurrence)
        # R[j_val] = R[k_val] + 1  (immediately after k_val's second occurrence)
        # This corresponds to the crossing interval pattern (L_k < L_j < R_k < R_j)
        # where L_j = L_k+1 and R_j = R_k+1.
        
        target_L_j = L_k + 1
        target_R_j = R_k + 1

        # Check if there's a value 'j_val' whose first occurrence is target_L_j
        j_val_from_L = first_pos_to_val.get(target_L_j)
        
        # Check if there's a value 'j_val' whose second occurrence is target_R_j
        j_val_from_R = second_pos_to_val.get(target_R_j)

        # If both lookups find a value and they are the same value (j_val)
        if j_val_from_L is not None and j_val_from_L == j_val_from_R:
            j_val = j_val_from_L
            
            # Ensure j_val is not a bad value (violates Condition 2)
            # Ensure k_val < j_val to count each pair (a, b) only once (a < b)
            if j_val not in bad_vals and k_val < j_val:
                count += 1
    
    sys.stdout.write(str(count) + "
")

# Read the number of test cases
num_test_cases = int(sys.stdin.readline())
for _ in range(num_test_cases):
    solve()