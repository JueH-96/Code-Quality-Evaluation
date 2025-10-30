import sys

def main():
    N = int(sys.stdin.readline())
    # P_list[k] stores P_{k+1} from the problem statement (0-indexed access for P_i).
    P_list = list(map(int, sys.stdin.readline().split()))

    # Segment tree setup
    # num_leaves: smallest power of 2 that is >= N. This is the number of leaf nodes
    #             in the segment tree structure.
    num_leaves = 1
    while num_leaves < N:
        num_leaves *= 2
    
    # The segment tree itself. It's 1-indexed for easier child/parent calculation.
    # tree[1] is the root. Children of tree[x] are tree[2*x] and tree[2*x+1].
    # Leaf nodes corresponding to original array indices 0...N-1 are stored at
    # tree[num_leaves]...tree[num_leaves + N - 1].
    # Total size is 2 * num_leaves. tree[0] is unused.
    tree = [0] * (2 * num_leaves)

    # Build phase:
    # Initialize leaves: 1 represents an empty slot.
    # Only initialize leaves that correspond to actual indices 0 to N-1.
    if N > 0: #This check is technically not needed due to constraints 1 <= N
        for i in range(N):
            tree[num_leaves + i] = 1
        # Initialize internal nodes by summing children values.
        # Iterate from the parent of the last potential leaf down to the root.
        for i in range(num_leaves - 1, 0, -1): # from num_leaves-1 down to 1
            tree[i] = tree[2 * i] + tree[2 * i + 1]

    ans = [0] * N  # Final array to store results

    # Process elements from N down to 1.
    # The input P_list is 0-indexed. P_list[k] corresponds to P_{k+1} for value (k+1).
    # So, iterate i_idx_in_P_list from N-1 down to 0.
    for i_idx_in_P_list in range(N - 1, -1, -1):
        # value_to_insert is N, then N-1, ..., down to 1.
        value_to_insert = i_idx_in_P_list + 1
        
        # target_rank is P_i (1-indexed) from problem spec.
        # For value_to_insert, its rank is P_list[i_idx_in_P_list].
        target_rank = P_list[i_idx_in_P_list]

        # Find the target_rank-th empty slot using segment tree traversal.
        # curr_node_idx starts at root (index 1).
        curr_node_idx = 1
        # Traverse until we reach a leaf node.
        # Leaf nodes are at indices [num_leaves, 2*num_leaves - 1].
        while curr_node_idx < num_leaves:
            left_child_sum = tree[2 * curr_node_idx]
            if target_rank <= left_child_sum: # Target is in left subtree
                curr_node_idx = 2 * curr_node_idx
            else: # Target is in right subtree
                target_rank -= left_child_sum # Adjust rank for right subtree
                curr_node_idx = 2 * curr_node_idx + 1
        
        # curr_node_idx is the segment tree index of the found leaf.
        # Convert this to 0-indexed position in the original conceptual array.
        # Leaf for original array pos `k` is at `num_leaves + k`.
        # So, original array pos `final_pos_0_indexed = curr_node_idx - num_leaves`.
        final_pos_0_indexed = curr_node_idx - num_leaves
        ans[final_pos_0_indexed] = value_to_insert

        # Update segment tree: mark this slot as filled (value 0).
        # The leaf to update is at curr_node_idx.
        tree[curr_node_idx] = 0
        # Propagate the change upwards to parent nodes.
        # Parent of node `x` is `x // 2`.
        p_idx = curr_node_idx // 2
        while p_idx > 0: # While not past the root
            tree[p_idx] = tree[2 * p_idx] + tree[2 * p_idx + 1]
            p_idx //= 2
            
    sys.stdout.write(" ".join(map(str, ans)) + "
")

if __name__ == '__main__':
    main()