import sys

def solve():
    input = sys.stdin.readline

    N, Q = map(int, input().split())
    A_list = list(map(int, input().split()))

    # We are interested in counts of numbers 0, ..., N.
    # counts array will be indexed from 0 to N, so its size is N+1.
    RELEVANT_VALUES_COUNT = N + 1 
    # Values in A_list or x_k that are > N are not relevant for mex if mex <= N.
    MAX_RELEVANT_VALUE = N
    
    # Use (N + 1) as infinity for segment tree values, as mex cannot exceed N.
    INFINITY = N + 1 

    counts = [0] * RELEVANT_VALUES_COUNT 
    for x_val in A_list:
        if x_val <= MAX_RELEVANT_VALUE:
            counts[x_val] += 1

    # Segment Tree Initialization
    seg_tree_actual_leaves = RELEVANT_VALUES_COUNT
    
    seg_tree_leaf_offset = 1
    while seg_tree_leaf_offset < seg_tree_actual_leaves:
        seg_tree_leaf_offset *= 2
    
    seg_tree = [INFINITY] * (2 * seg_tree_leaf_offset)

    for i in range(seg_tree_actual_leaves): # i from 0 to N
        if counts[i] == 0:
            seg_tree[seg_tree_leaf_offset + i] = i
        else:
            seg_tree[seg_tree_leaf_offset + i] = INFINITY 
            
    for i in range(seg_tree_leaf_offset - 1, 0, -1):
        seg_tree[i] = min(seg_tree[2 * i], seg_tree[2 * i + 1])

    def update_seg_tree(idx_to_update, new_leaf_val):
        pos = seg_tree_leaf_offset + idx_to_update
        seg_tree[pos] = new_leaf_val
        
        pos //= 2 
        while pos >= 1:
            seg_tree[pos] = min(seg_tree[2 * pos], seg_tree[2 * pos + 1])
            pos //= 2

    results = []
    for _ in range(Q):
        i_k, x_k_new_val = map(int, input().split())
        idx_in_A = i_k - 1 

        old_val = A_list[idx_in_A]
        A_list[idx_in_A] = x_k_new_val 
        
        if old_val <= MAX_RELEVANT_VALUE:
            counts[old_val] -= 1
            if counts[old_val] == 0:
                update_seg_tree(old_val, old_val)
        
        if x_k_new_val <= MAX_RELEVANT_VALUE:
            # Check if its count was zero BEFORE incrementing counts[x_k_new_val]
            # This is equivalent to (counts[x_k_new_val] + 1 == 1) after incrementing.
            # Only update seg_tree if it *was* a mex candidate (count 0)
            # and now it's not (count becomes 1).
            # If count was already >=1, it remains not a candidate.
            
            # Store previous count state to check if it was 0
            # is_zero_before_inc = (counts[x_k_new_val] == 0) 
            # counts[x_k_new_val] += 1
            # if is_zero_before_inc: # True if counts[x_k_new_val] is now 1 and was 0
            #    update_seg_tree(x_k_new_val, INFINITY)
            
            # Simpler logic: increment, then check if count became 1.
            # This correctly identifies a 0 -> 1 transition.
            counts[x_k_new_val] += 1
            if counts[x_k_new_val] == 1: 
                update_seg_tree(x_k_new_val, INFINITY)
        
        results.append(str(seg_tree[1]))

    sys.stdout.write("
".join(results) + "
")

if __name__ == '__main__':
    solve()