import sys

class Solution:
    def longestSpecialPath(self, edges: List[List[int]], nums: List[int]) -> List[int]:
        n = len(nums)
        
        # Constraints: n >= 2. Path of single node is length 0, 1 node.
        # Default answer if no multi-node special path exists or all are length 0.
        max_len_res = 0
        min_nodes_res = 1 

        # Python recursion limit can be an issue for deep trees.
        # Max depth is n. Default limit (e.g., 1000 or 3000) might not be enough.
        sys.setrecursionlimit(n + 100) # Increased recursion limit

        adj = [[] for _ in range(n)]
        for u_node, v_node, length in edges:
            adj[u_node].append((v_node, length))
            adj[v_node].append((u_node, length))

        # Segment Tree setup
        # Max depth is n-1. Depths range 0 to n-1.
        # We map depth d to segment tree index d+1 (1-based for depths).
        # A dummy depth -1 (st_idx 0) helps handle paths starting at root.
        # So segment tree indices effectively cover depths -1 to n-1, total n+1 distinct depths.
        
        # Determine size for segment tree array (power of 2)
        # Max st_idx will be n (for depth n-1). Min st_idx will be 0 (for depth -1).
        # So we need to cover indices 0 to n.
        st_leaf_level_size = 1
        while st_leaf_level_size <= n: # Max index is n (for depth n-1)
            st_leaf_level_size *= 2
        
        # Segment tree stores (L_val, d_val) pairs.
        # Query goal: min L_val, then max d_val.
        # Neutral element for query: (infinity L_val, -infinity d_val)
        st_neutral = (float('inf'), float('-inf'))
        # Segment tree array, 1-indexed for tree structure, 0 unused.
        st_data = [st_neutral] * (2 * st_leaf_level_size)

        def st_update(original_depth_idx, L_val, d_val):
            # original_depth_idx is the actual depth (-1 to n-1)
            # mapped_st_idx is segment tree's leaf index (0 to n)
            mapped_st_idx = original_depth_idx + 1 
            
            pos = st_leaf_level_size + mapped_st_idx # Position in st_data array
            st_data[pos] = (L_val, d_val)
            
            while pos > 1: # Propagate update upwards
                pos //= 2
                left_child_val, left_child_d = st_data[pos * 2]
                right_child_val, right_child_d = st_data[pos * 2 + 1]
                
                if left_child_val < right_child_val:
                    st_data[pos] = (left_child_val, left_child_d)
                elif right_child_val < left_child_val:
                    st_data[pos] = (right_child_val, right_child_d)
                else: # L_vals are equal, pick one with larger d_val (max depth)
                    if left_child_d > right_child_d:
                        st_data[pos] = (left_child_val, left_child_d)
                    else:
                        st_data[pos] = (right_child_val, right_child_d)
        
        def st_query(range_start_orig_depth, range_end_orig_depth):
            # Query on interval of original depths [range_start_orig_depth, range_end_orig_depth]
            if range_start_orig_depth > range_end_orig_depth:
                return st_neutral # Empty range

            l_st_idx = range_start_orig_depth + 1 # Map to st_idx
            r_st_idx = range_end_orig_depth + 1   # Map to st_idx
            
            # Actual query on segment tree leaf indices
            l = st_leaf_level_size + l_st_idx
            r = st_leaf_level_size + r_st_idx
            
            res_L, res_d = st_neutral
            while l <= r:
                if l % 2 == 1: # l is right child of its parent
                    cand_L, cand_d = st_data[l]
                    if cand_L < res_L:
                        res_L, res_d = cand_L, cand_d
                    elif cand_L == res_L and cand_d > res_d:
                        res_L, res_d = cand_L, cand_d
                    l += 1
                if r % 2 == 0: # r is left child of its parent
                    cand_L, cand_d = st_data[r]
                    if cand_L < res_L:
                        res_L, res_d = cand_L, cand_d
                    elif cand_L == res_L and cand_d > res_d:
                        res_L, res_d = cand_L, cand_d
                    r -= 1
                l //= 2
                r //= 2
            return res_L, res_d

        # Stores depth of node with value `val` on current path from root
        val_to_depth_on_curr_path = {} 

        # Initialize segment tree with dummy ancestor for root
        # Depth -1, Path Length 0. This allows paths starting at root to be formed.
        st_update(-1, 0, -1)

        # u: current node
        # p: parent of u in DFS traversal
        # depth_u: depth of u (root is 0)
        # path_L_u: path length from root (node 0) to u
        def dfs_solve(u, p, depth_u, path_L_u):
            nonlocal max_len_res, min_nodes_res

            # Determine limit_anc_depth: depth of the deepest ancestor 'x' of u (could be on path 0...parent(u))
            # such that nums[x] == nums[u]. If no such ancestor, -1.
            limit_anc_depth = val_to_depth_on_curr_path.get(nums[u], -1)
            
            # Query for best ancestor 'a'. 'a' must be at depth d_a.
            # Query range for d_a: [limit_anc_depth + 1, depth_u]
            # (ancestor 'a' can be 'u' itself)
            query_start_depth = limit_anc_depth + 1
            query_end_depth = depth_u
            
            # L_best_a: min path length from root to an ancestor 'a' in query range
            # d_best_a: max depth of such 'a' (if L_best_a ties)
            L_best_a, d_best_a = st_query(query_start_depth, query_end_depth)

            if L_best_a != float('inf'): # Check if a valid ancestor was found
                cand_L = path_L_u - L_best_a
                cand_N = depth_u - d_best_a + 1
                
                if cand_L > max_len_res:
                    max_len_res = cand_L
                    min_nodes_res = cand_N
                elif cand_L == max_len_res:
                    if cand_N < min_nodes_res:
                        min_nodes_res = cand_N
            
            # Store current state for backtracking
            prev_val_depth_entry = val_to_depth_on_curr_path.get(nums[u], None)
            # Mapped st_idx for current depth_u
            # prev_st_val_L, prev_st_val_d = st_query(depth_u, depth_u) This is slow, just save the specific old value if needed, or clear.
            # Since each depth on path is unique, we are overwriting. We just need to clear it on backtrack.
            
            # Update state for processing u's children
            val_to_depth_on_curr_path[nums[u]] = depth_u
            st_update(depth_u, path_L_u, depth_u) 

            for v_node, length_uv in adj[u]:
                if v_node == p: # Don't go back to parent
                    continue
                dfs_solve(v_node, u, depth_u + 1, path_L_u + length_uv)

            # Backtrack: Restore state
            if prev_val_depth_entry is None:
                del val_to_depth_on_curr_path[nums[u]]
            else:
                val_to_depth_on_curr_path[nums[u]] = prev_val_depth_entry
            
            # Remove/neutralize u's entry from segment tree
            st_update(depth_u, st_neutral[0], st_neutral[1])


        dfs_solve(0, -1, 0, 0) # Start DFS from root (node 0)
        
        return [max_len_res, min_nodes_res]