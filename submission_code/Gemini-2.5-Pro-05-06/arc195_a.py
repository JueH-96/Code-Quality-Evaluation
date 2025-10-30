import collections
import bisect
import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    A_list = list(map(int, sys.stdin.readline().split())) # Renamed to A_list to avoid conflict
    B_list = list(map(int, sys.stdin.readline().split())) # Renamed to B_list

    locs = collections.defaultdict(list)
    for i, val in enumerate(A_list):
        locs[val].append(i)

    # Compute L_indices (leftmost subsequence indices)
    # L_indices[j] stores the index in A_list used for B_list[j]
    L_indices = [-1] * M
    current_A_idx_L = -1 # Last used index in A_list, must pick indices > this
    for j in range(M):
        val_B = B_list[j]
        if val_B not in locs: # Value B_list[j] does not exist in A_list
            print("No")
            return
        
        possible_indices_for_val_B = locs[val_B]
        
        # Find smallest index k_ptr_val in possible_indices_for_val_B 
        # such that possible_indices_for_val_B[k_ptr_val] > current_A_idx_L
        # k_ptr_idx is the index within possible_indices_for_val_B
        k_ptr_idx = bisect.bisect_right(possible_indices_for_val_B, current_A_idx_L)
        
        if k_ptr_idx == len(possible_indices_for_val_B):
            # No index in A_list found for B_list[j] that is > current_A_idx_L
            print("No")
            return
        
        L_indices[j] = possible_indices_for_val_B[k_ptr_idx]
        current_A_idx_L = L_indices[j]

    # Compute R_indices (rightmost subsequence indices)
    # R_indices[j] stores the index in A_list used for B_list[j]
    R_indices = [-1] * M
    current_A_idx_R = N # Last used index in A_list (conceptually, from right end), must pick indices < this
    for j in range(M - 1, -1, -1):
        val_B = B_list[j]
        # val_B must be in locs if L_indices computation succeeded
        
        possible_indices_for_val_B = locs[val_B]
        
        # Find largest index k_ptr_val in possible_indices_for_val_B 
        # such that possible_indices_for_val_B[k_ptr_val] < current_A_idx_R
        # bisect.bisect_left returns an insertion point k_ptr_idx such that all elements 
        # in possible_indices_for_val_B[:k_ptr_idx] are < current_A_idx_R,
        # and all elements in possible_indices_for_val_B[k_ptr_idx:] are >= current_A_idx_R.
        # So, possible_indices_for_val_B[k_ptr_idx-1] is the largest element < current_A_idx_R, if k_ptr_idx > 0.
        k_ptr_idx = bisect.bisect_left(possible_indices_for_val_B, current_A_idx_R)

        if k_ptr_idx == 0:
            # No index in A_list found for B_list[j] that is < current_A_idx_R
            # This implies B_list is not a subsequence if reached, though L_indices check should catch it.
            print("No") 
            return
        
        R_indices[j] = possible_indices_for_val_B[k_ptr_idx-1]
        current_A_idx_R = R_indices[j]
        
    # If L_indices and R_indices are different, they represent two distinct subsequences.
    if L_indices != R_indices:
        print("Yes")
    else:
        # If L_indices == R_indices, this specific set of indices is the
        # only way to form B_list as a subsequence of A_list.
        print("No")

solve()