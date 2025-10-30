# YOUR CODE HERE
import sys
import bisect
from collections import defaultdict

def solve():
    # Read N (length of sequence A) and M (length of sequence B) from input
    N, M = map(int, sys.stdin.readline().split())
    
    # Read sequence A
    A = list(map(int, sys.stdin.readline().split()))
    
    # Read sequence B
    B = list(map(int, sys.stdin.readline().split()))

    # Precompute indices for each value in A using a dictionary (hash map).
    # The keys are the distinct values present in A.
    # The values are sorted lists containing the 0-based indices where each value appears in A.
    # Using defaultdict simplifies adding indices to lists.
    val_to_indices = defaultdict(list)
    for i in range(N):
        val_to_indices[A[i]].append(i)

    # --- Compute the leftmost subsequence indices L ---
    # L[k] will store the smallest possible index `i_k` in A such that
    # A[i_k] = B[k] and there exists a sequence of indices i_1 < ... < i_{k-1} < i_k
    # matching B[0...k]. This is found by always picking the minimal possible index at each step.
    L = [0] * M  # Initialize L array of size M
    # `current_idx` tracks the index used for the previous element B[k-1].
    # Initialized to -1, representing a position before the start of A.
    current_idx = -1  
    
    # Iterate through B from k=0 to M-1
    for k in range(M):
        target_val = B[k] # The value we need to find in A
        
        # Check if the required value B[k] exists in A at all.
        if target_val not in val_to_indices:
            # If B[k] is not present in A, no subsequence matching B can exist.
            print("No")
            return

        # Get the sorted list of indices where target_val appears in A
        idx_list = val_to_indices[target_val]
        
        # We need to find the smallest index `idx` in `idx_list` such that `idx > current_idx`.
        # `bisect.bisect_right(idx_list, current_idx)` finds the insertion point `pos` for `current_idx`
        # such that all elements `idx_list[j]` with `j >= pos` satisfy `idx_list[j] > current_idx`.
        # Therefore, `idx_list[pos]` (if it exists) is the smallest element in `idx_list` 
        # that is strictly greater than `current_idx`.
        pos = bisect.bisect_right(idx_list, current_idx)
        
        # Check if such an index exists
        if pos == len(idx_list):
            # If `pos` is equal to the length of `idx_list`, it means no element in `idx_list` 
            # is strictly greater than `current_idx`.
            # Thus, we cannot extend the subsequence matching B with B[k] using an index after `current_idx`.
            print("No")
            return
        
        # The found index `idx_list[pos]` is the smallest valid index for B[k]
        # in the context of building the leftmost subsequence.
        found_idx = idx_list[pos]
        L[k] = found_idx # Store this index in the L array
        # Update `current_idx` to the index just chosen, for the constraint of the next element B[k+1].
        current_idx = found_idx

    # --- Compute the rightmost subsequence indices R ---
    # R[k] will store the largest possible index `i_k` in A such that
    # A[i_k] = B[k] and there exists a sequence of indices i_k < i_{k+1} < ... < i_M
    # matching B[k...M-1]. This is found by always picking the maximal possible index at each step.
    R = [0] * M # Initialize R array of size M
    # `current_idx` tracks the index used for the next element B[k+1].
    # Initialized to N, representing a position after the end of A.
    current_idx = N  
    
    # Iterate backwards through B from k=M-1 down to 0
    for k in range(M - 1, -1, -1):
        target_val = B[k] # The value we need to find in A
        
        # This check is technically redundant if L computation succeeded, 
        # because if any B[k] was missing, L computation would have already returned "No".
        # Including it doesn't harm and adds robustness.
        if target_val not in val_to_indices:
             print("No") # Should realistically not be reached if L computation succeeded for M > 0.
             return

        # Get the sorted list of indices where target_val appears in A
        idx_list = val_to_indices[target_val]
        
        # We need to find the largest index `idx` in `idx_list` such that `idx < current_idx`.
        # `bisect.bisect_left(idx_list, current_idx)` finds the insertion point `pos` for `current_idx`
        # such that all elements `idx_list[i]` with `i < pos` satisfy `idx_list[i] < current_idx`.
        # Therefore, `idx_list[pos-1]` (if `pos > 0`) is the largest element in `idx_list` 
        # strictly less than `current_idx`.
        pos = bisect.bisect_left(idx_list, current_idx)
        
        # Check if such an index exists
        if pos == 0:
            # If `pos` is 0, it means no element in `idx_list` is strictly less than `current_idx`.
            # This implies all occurrences of target_val are >= current_idx.
            # We cannot find a suitable index for B[k] preceding the element at `current_idx`.
            print("No") 
            return

        # The found index `idx_list[pos-1]` is the largest valid index for B[k]
        # in the context of building the rightmost subsequence.
        found_idx = idx_list[pos - 1]
        R[k] = found_idx # Store this index in the R array
        # Update `current_idx` to the index just chosen, for the constraint of the previous element B[k-1].
        current_idx = found_idx

    # --- Final Determination ---
    # Compare the computed leftmost index sequence L and rightmost index sequence R.
    # If L and R are identical, it implies that for every element B[k], the choice of index in A 
    # was forced to be the same value for both the minimal (leftmost) and maximal (rightmost) constructions. 
    # This indicates there is only one unique sequence of indices that matches B.
    # If L and R differ at any position k (i.e., L[k] != R[k]), it means there was flexibility 
    # in choosing the index for B[k] (or some other element), allowing for at least two distinct 
    # sequences of indices. The leftmost and rightmost sequences themselves are two such distinct subsequences.
    if L == R:
        # If L equals R, there is only one subsequence matching B.
        print("No") 
    else:
        # If L is different from R, there are at least two distinct subsequences matching B.
        print("Yes") 

# Call the solve function to execute the program logic when the script is run
solve()