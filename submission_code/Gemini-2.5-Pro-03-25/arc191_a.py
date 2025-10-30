# YOUR CODE HERE
import sys

# Set higher recursion depth limit for deep trees potentially needed for N=10^6
# Use try-except block as setting recursion depth limit might fail in some environments
try:
    # A large enough value, adjust if needed based on constraints and typical Python limits
    # N, M <= 10^6, segment tree depth is log N, roughly 20. Recursion depth should be okay,
    # but setting it higher is safer for potentially deep recursive calls.
    sys.setrecursionlimit(2 * 10**6 + 10) 
except Exception as e: 
    # If setting limit fails, print error message and continue, maybe it's sufficient?
    # print(f"Warning: Failed to set recursion depth limit. {e}", file=sys.stderr)
    pass 

def solve():
    # Read N and M from input
    N, M = map(int, sys.stdin.readline().split())
    # Read string S and convert to list of characters
    S_list = list(sys.stdin.readline().strip())
    # Read string T and convert to list of characters
    T_list = list(sys.stdin.readline().strip())

    # Convert characters '1'-'9' to integers 1-9 for easier comparison and processing
    S_val = [int(c) for c in S_list]
    T_val = [int(c) for c in T_list]

    # Handle edge case N=0, although constraints state N >= 1
    if N == 0:
        print()
        return

    # Determine the size needed for the segment tree array. 4*N is a safe upper bound.
    tree_nodes = 4 * N
    # Initialize segment tree nodes with a value larger than any possible digit (1-9). 
    # 10 works as a sentinel value indicating 'empty' or 'no digit yet'.
    min_val_tree = [10] * tree_nodes 
    
    # This array will store the current values at each position of string S.
    # It's updated alongside the segment tree. The segment tree helps find the target index efficiently.
    # `current_s_val` is used to reconstruct the final string easily at the end.
    current_s_val = list(S_val) 

    # Function to build the segment tree initially based on the initial string S_val
    def build_min_val(v, tl, tr):
        """
        Builds the segment tree recursively.
        v: current node index (1-based)
        tl: starting index of the range covered by node v
        tr: ending index of the range covered by node v
        """
        # Base case: leaf node representing a single index
        if tl == tr:
             # Check array bounds: leaf index tl must be within [0, N-1]
             if tl < N: 
                 min_val_tree[v] = S_val[tl]
             # If tl >= N, this node might exist due to tree structure (if N is not power of 2)
             # It corresponds to an index outside the actual string length. Keep it initialized to 10.
        else:
            # Recursive case: internal node
            tm = (tl + tr) // 2 # Find midpoint of the range [tl, tr]
            # Build left child (covers [tl, tm]) and right child (covers [tm+1, tr]) recursively
            build_min_val(v * 2, tl, tm)
            build_min_val(v * 2 + 1, tm + 1, tr)
            # Combine results: the minimum value in node v's range is the minimum of its children's ranges
            min_val_tree[v] = min(min_val_tree[v * 2], min_val_tree[v * 2 + 1])
    
    # Start building the tree from the root (node 1) covering the full index range [0, N-1]
    # Check if N > 0 before building to avoid issues with empty range if N=0 was possible.
    if N > 0:
        build_min_val(1, 0, N-1)

    # Function to update the value at a specific position `pos` to `new_val`
    # This updates the leaf node and propagates the change up the tree.
    def update_min_val(v, tl, tr, pos, new_val):
        """
        Updates the segment tree after changing a value at a specific position.
        v: current node index
        tl, tr: range covered by node v
        pos: index in the original array S to update
        new_val: the new digit value
        """
        # Base case: reached the leaf node corresponding to `pos`
        if tl == tr:
            min_val_tree[v] = new_val
        else:
            # Recursive case: internal node
            tm = (tl + tr) // 2 # Find midpoint
            if pos <= tm:
                # If position `pos` is in the left child's range, update the left child
                update_min_val(v * 2, tl, tm, pos, new_val)
            else:
                # If position `pos` is in the right child's range, update the right child
                update_min_val(v * 2 + 1, tm + 1, tr, pos, new_val)
            # After updating the relevant child, recalculate the minimum value for the current node `v`
            min_val_tree[v] = min(min_val_tree[v * 2], min_val_tree[v * 2 + 1])

    # Query function to find the minimum index `j` such that the value at `j` is less than `d`
    # The value at `j` is implicitly tracked by the `min_val_tree` state.
    def find_min_idx_less_than(v, tl, tr, d):
        """
        Finds the minimum index j in range [tl, tr] such that the value S[j] < d.
        Uses the segment tree `min_val_tree` for efficient searching.
        Returns the minimum index j if found, otherwise returns -1.
        v: current node index
        tl, tr: range covered by node v
        d: the digit value to compare against
        """
        # Pruning optimization: If the minimum value in the current node's range `[tl, tr]`
        # is already greater than or equal to `d`, then no element in this range can possibly
        # satisfy the condition `value < d`. Return -1 immediately.
        if min_val_tree[v] >= d: 
            return -1 
        
        # Base case: Reached a leaf node (tl == tr). 
        # Since the initial check `min_val_tree[v] >= d` passed, this leaf's value must be < d.
        # Return its index `tl`.
        if tl == tr: 
             return tl
        
        tm = (tl + tr) // 2 # Find midpoint
        
        # Recursive step: Prioritize searching the left child first (range [tl, tm]).
        # This is because we are looking for the minimum index `j`. If a qualifying index exists
        # in the left subtree, it must be smaller than any qualifying index in the right subtree.
        res_left = find_min_idx_less_than(v * 2, tl, tm, d)
        if res_left != -1:
             # If found in the left subtree, return the result immediately.
             return res_left 
        
        # If not found in the left subtree (res_left == -1), search the right subtree (range [tm+1, tr]).
        # If found here, it will be the minimum index overall since the left side had none.
        return find_min_idx_less_than(v * 2 + 1, tm + 1, tr, d)
    
    
    # Process each operation k from 0 to M-1 (corresponding to k=1 to M in problem statement)
    for k in range(M):
        # Get the digit `d` for the current operation T[k]
        d = T_val[k]
        
        # Use the segment tree query to find the minimum index `target_idx` such that the current value
        # at that index in S (as reflected in the segment tree state) is strictly less than `d`.
        target_idx = find_min_idx_less_than(1, 0, N-1, d)

        if target_idx != -1:
            # If such an index `target_idx` is found:
            # This is the position where replacing the current digit with `d` gives the
            # largest lexicographical improvement (most significant improvement).
            # Update the segment tree at this position `target_idx` with the new digit `d`.
            update_min_val(1, 0, N-1, target_idx, d)
            # Also update the auxiliary array `current_s_val` that tracks the current state of S.
            current_s_val[target_idx] = d 
        else:
             # If no index `j` is found where `S[j] < d`:
             # This means the digit `d` is less than or equal to all current digits in `S`.
             # Replacing any digit `S[j]` with `d` will either keep the string same (if S[j]=d) 
             # or make it lexicographically smaller (if S[j]>d).
             # The chosen greedy strategy in this case is to replace the digit at the least significant 
             # position, N-1, to minimize the negative impact on the overall integer value.
             target_idx = N - 1 # Choose the last index
             # Update the segment tree at index N-1 with the new digit `d`.
             update_min_val(1, 0, N-1, target_idx, d)
             # Update the auxiliary array `current_s_val`.
             current_s_val[target_idx] = d 

    # After processing all M operations, `current_s_val` holds the final digits of string S.
    # Convert the list of integers back to a string by joining them.
    print("".join(map(str, current_s_val)))

# Call the main solving function to execute the logic
solve()