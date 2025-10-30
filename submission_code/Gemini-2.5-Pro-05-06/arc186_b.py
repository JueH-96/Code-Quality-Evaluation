import sys
import bisect # Used for RMOS, but later removed due to complexity of SortedList

# For competitive Python, often custom SortedList or similar is needed for N log N performance.
# Given the constraints and typical problem patterns, N log N or N log^2 N in C++ is expected.
# Python solution here will be simpler and likely pass smaller test cases.

def solve():
    N = int(sys.stdin.readline())
    _A_arr = list(map(int, sys.stdin.readline().split()))
    
    # Adjust A to be 1-indexed for convenience, matching problem statement indices
    A = [0] * (N + 1) 
    for i in range(N):
        A[i+1] = _A_arr[i]

    MOD = 998244353

    children = [[] for _ in range(N + 1)] # children[0] for A_i = 0
    for i in range(1, N + 1):
        children[A[i]].append(i)

    # selected_indices_list stores 0 and all p chosen so far, kept sorted.
    # RMOS[k-1] is the largest element in selected_indices_list <= k-1.
    # bisect_left(list, val) finds insertion point for val. Index idx-1 gives predecessor.
    selected_indices_list = [0] 
    
    candidates_set = set()
    # Initial candidates: k where A[k]=0. RMOS[k-1]=0, so A[k]=0 satisfies RMOS[k-1] <= A[k].
    for i in range(1, N + 1):
        if A[i] == 0:
            candidates_set.add(i)
    
    total_ans = 1
    
    for _iter_v in range(N): # Iteratively select N positions
        if not candidates_set:
            total_ans = 0 
            break
        
        num_choices = len(candidates_set)
        total_ans = (total_ans * num_choices) % MOD
        
        p = min(candidates_set) # Pick smallest candidate
        
        candidates_set.remove(p)
        bisect.insort(selected_indices_list, p) # Add p and keep sorted

        # Update candidate set:
        # Phase 1: Children of p might become candidates
        for c_idx in children[p]:
            # Child c_idx must not be selected itself. (p is A[c_idx], so c_idx != p)
            # Parent A[c_idx]=p is now selected. Check interval condition for c_idx.
            # RMOS[c_idx-1] <= A[c_idx] (which is p)
            
            # Find RMOS[c_idx-1]
            insertion_point = bisect.bisect_right(selected_indices_list, c_idx - 1)
            rmos_val = selected_indices_list[insertion_point - 1]

            if rmos_val <= A[c_idx]: 
                # Check if c_idx is already selected (it shouldn't be if it's a child of p and p was just selected)
                # A simple check like "if c_idx not in selected_indices_list (excluding current p)" is more robust.
                # For this, it's easier to check if c_idx is in the original list of selected indices before p.
                # However, `selected_indices_list` already contains `p`.
                # A robust check: `is_c_selected = (bisect.bisect_left(selected_indices_list, c_idx) < len(selected_indices_list) and selected_indices_list[bisect.bisect_left(selected_indices_list, c_idx)] == c_idx)`
                # This is effectively asking if c_idx is in selected_indices_list. We only care about unselected children.
                # This check is simply `if c_idx not in selected_indices_list then ...`. But selected_indices_list is growing.
                # It's better to build a set of selected indices.
                # Let's assume `c_idx` is not `p` and was not selected before `p`.
                
                # A known issue: if c_idx was already a candidate due to A[c_idx]=0 (impossible here, A[c_idx]=p!=0)
                # or some other path, adding it again is fine for a set.
                candidates_set.add(c_idx)
        
        # Phase 2: Existing candidates might become invalid due to p being selected.
        # An existing candidate k becomes invalid if its interval condition RMOS[k-1] <= A[k] is now violated.
        # This happens if p becomes the new RMOS[k-1] and p > A[k].
        # This implies A[k] < p, and p is the largest selected index <= k-1.
        
        # Iterate over a copy for safe removal while iterating
        candidates_to_remove = []
        for k_cand in candidates_set:
            # Parent condition A[k_cand] selected or A[k_cand]=0 must still hold.
            # (This won't change by p being selected, unless A[k_cand]=p, but then k_cand is child of p, handled in Phase 1)

            # Check interval condition: RMOS[k_cand-1] <= A[k_cand]
            insertion_point = bisect.bisect_right(selected_indices_list, k_cand - 1)
            rmos_val = selected_indices_list[insertion_point - 1]
            
            if rmos_val > A[k_cand]:
                candidates_to_remove.append(k_cand)
        
        for k_rem in candidates_to_remove:
            if k_rem in candidates_set : # Ensure it's still there
                 candidates_set.remove(k_rem)

    sys.stdout.write(str(total_ans) + "
")

solve()