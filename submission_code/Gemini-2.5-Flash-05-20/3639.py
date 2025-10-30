import collections

class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        N = len(nums)
        
        # We need 4*N size for segment tree arrays
        # tree[idx] stores [min_val, max_val] in its range
        tree = [[0, 0] for _ in range(4 * N)]  
        # lazy[idx] stores the pending decrement to be applied to tree[idx]'s range
        lazy = [0] * (4 * N)  
        
        # Build the segment tree
        # idx: current node index in tree array
        # L, R: range covered by current node [L, R]
        def build(idx, L, R):
            if L == R:
                # Leaf node: stores the value of nums[L]
                tree[idx] = [nums[L], nums[L]]
                return
            
            mid = (L + R) // 2
            build(2 * idx, L, mid)          # Build left child
            build(2 * idx + 1, mid + 1, R)  # Build right child
            
            # Internal node: min_val is min of children's min_val, max_val is max of children's max_val
            tree[idx][0] = min(tree[2 * idx][0], tree[2 * idx + 1][0])
            tree[idx][1] = max(tree[2 * idx][1], tree[2 * idx + 1][1])

        # Push down lazy tags from current node to its children
        def push_down(idx):
            # If there's a pending update (lazy tag is not 0)
            if lazy[idx] != 0:
                # Apply lazy tag to children's min_val and max_val
                for child_idx in [2 * idx, 2 * idx + 1]:
                    tree[child_idx][0] += lazy[idx]
                    tree[child_idx][1] += lazy[idx]
                    lazy[child_idx] += lazy[idx]    # Propagate lazy tag to children
                lazy[idx] = 0 # Clear lazy tag for current node
        
        # Update range [queryL, queryR] by adding 'val' (which will be -1 for decrement)
        # idx: current node index
        # L, R: range covered by current node [L, R]
        # queryL, queryR: range for the update query
        # val: value to apply (e.g., -1)
        def update(idx, L, R, queryL, queryR, val):
            # Case 1: No overlap between current node's range and query range
            if R < queryL or L > queryR:
                return
            
            # Case 2: Full overlap - current node's range is completely within query range
            if queryL <= L and R <= queryR:
                # If minimum value in range is 0, we cannot simply apply lazy tag
                # because decrementing 0 would make it negative.
                # E.g., if tree[idx][0] is 0 and val is -1, then tree[idx][0] + val < 0.
                if tree[idx][0] + val < 0:
                    # If it's a leaf node and its value is 0 (or would become negative), clamp it to 0
                    if L == R:
                        tree[idx][0] = tree[idx][1] = 0 # Value stays 0
                        return
                    else:
                        # For internal nodes where min_val is 0, we must descend.
                        # Some elements might be >0 and can be decremented, others are 0 and cannot.
                        push_down(idx) # Push down any pending lazy tag before descending
                        mid = (L + R) // 2
                        update(2 * idx, L, mid, queryL, queryR, val)      # Update left child
                        update(2 * idx + 1, mid + 1, R, queryL, queryR, val) # Update right child
                        # Recalculate min/max for current node based on children's updated values
                        tree[idx][0] = min(tree[2 * idx][0], tree[2 * idx + 1][0])
                        tree[idx][1] = max(tree[2 * idx][1], tree[2 * idx + 1][1])
                        return
                else: # Case where tree[idx][0] + val >= 0 (i.e., min_val is >= 1, so all values in range are >= 1)
                      # Safe to decrement all values in this range directly
                    tree[idx][0] += val # Apply val to min_val
                    tree[idx][1] += val # Apply val to max_val
                    lazy[idx] += val    # Add val to lazy tag
                    return
            
            # Case 3: Partial overlap - current node's range partially overlaps query range
            # Descend to children
            push_down(idx) # Push down any pending lazy tag before descending
            mid = (L + R) // 2
            update(2 * idx, L, mid, queryL, queryR, val)      # Update left child
            update(2 * idx + 1, mid + 1, R, queryL, queryR, val) # Update right child
            
            # Recalculate min/max for current node based on children's updated values
            tree[idx][0] = min(tree[2 * idx][0], tree[2 * idx + 1][0])
            tree[idx][1] = max(tree[2 * idx][1], tree[2 * idx + 1][1])

        # 1. Initial build of the segment tree
        build(1, 0, N - 1)
        
        # 2. Process queries sequentially
        for l, r in queries:
            update(1, 0, N - 1, l, r, -1) # Decrement by 1 for all selected indices in range [l, r]
        
        # 3. After all queries, check if the entire array has become a Zero Array
        # This is true if and only if the maximum value in the entire array is 0.
        # The root node (idx=1) covers the entire array [0, N-1], so we check tree[1][1] (max_val).
        return tree[1][1] == 0