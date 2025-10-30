import sys

def solve():
    N = int(sys.stdin.readline())
    # Input A_1 ... A_N are values from 1 to N.
    # Read them into a 0-indexed list `A`.
    A_str_list = sys.stdin.readline().split()
    A = [int(x) for x in A_str_list]

    # val_to_idx[value] stores the 0-indexed position of 'value' in A.
    # Values are 1...N. We make val_to_idx 1-indexed for values (val_to_idx[0] unused).
    val_to_idx = [0] * (N + 1)
    for i in range(N):
        val_to_idx[A[i]] = i  # A[i] is the value, i is its 0-indexed position
    
    swaps_list = []

    # Standard selection sort: iterate i from 0 to N-2.
    # The element A[N-1] (last one) will be in place if A[0]...A[N-2] are correct.
    # The loop `for i in range(N-1)` correctly gives indices 0, 1, ..., N-2.
    for i in range(N - 1): # This loop runs N-1 times, for i = 0, 1, ..., N-2
        # The value that should be at index i is (i+1) (since values are 1-N).
        target_val_for_i = i + 1
        
        # Check if the correct value is already at A[i]
        if A[i] == target_val_for_i:
            continue  # Element is already in its correct place
        else:
            # A[i] is not target_val_for_i. We need to bring target_val_for_i to A[i].
            # Find where target_val_for_i is currently located.
            # Its 0-indexed position is val_to_idx[target_val_for_i].
            idx_of_target_val = val_to_idx[target_val_for_i]
            
            # The value currently at A[i] will be moved to A[idx_of_target_val].
            current_val_at_i = A[i]
            
            # Perform swap in array A:
            # A[i] gets target_val_for_i, A[idx_of_target_val] gets current_val_at_i.
            A[i], A[idx_of_target_val] = A[idx_of_target_val], A[i]
            # After swap: A[i] IS target_val_for_i
            #              A[idx_of_target_val] IS current_val_at_i
            
            # Update val_to_idx for the two values that moved their positions.
            # target_val_for_i is now at index i.
            val_to_idx[target_val_for_i] = i
            # current_val_at_i is now at index idx_of_target_val.
            val_to_idx[current_val_at_i] = idx_of_target_val
            
            # Record swap using 1-based indices for positions.
            # Output positions must be p1, p2 such that 1 <= p1 < p2 <= N.
            # Current 0-indexed position is i. 1-based is i+1.
            # Target value's original 0-indexed position was idx_of_target_val. 1-based is idx_of_target_val+1.
            # If a swap is needed, target_val_for_i is not at A[i]. Since elements A[0]...A[i-1] are
            # already correct (i.e. hold values 1...i), target_val_for_i must be at an index > i.
            # So, idx_of_target_val > i. This means (i+1) < (idx_of_target_val+1).
            pos1 = i + 1
            pos2 = idx_of_target_val + 1
            swaps_list.append((pos1, pos2))

    # Output results
    sys.stdout.write(str(len(swaps_list)) + "
")
    for p1, p2 in swaps_list:
        sys.stdout.write(f"{p1} {p2}
")

# This structure is for competitive programming where the function is called once.
if __name__ == '__main__':
    solve()