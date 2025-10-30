import sys

# It might be necessary to increase recursion limit for a recursive segment tree,
# but an iterative one is used here.
# sys.setrecursionlimit(2 * 10**5) 

class Solution:
  def longestCommonPrefix(self, words: list[str], k: int) -> list[int]:
    n = len(words)
    ans = [0] * n

    if k == 0: # According to constraints 1 <= k, but being defensive.
        return ans 
    
    if n - 1 < k:
        return ans

    # Trie construction
    trie_nodes = [{'children': {}, 'word_indices': [], 'depth': 0}] # Root node

    for i, word in enumerate(words):
        curr_node_idx = 0
        trie_nodes[curr_node_idx]['word_indices'].append(i) # Add to root
        for char_idx, char in enumerate(word):
            if char not in trie_nodes[curr_node_idx]['children']:
                new_node_idx = len(trie_nodes)
                trie_nodes[curr_node_idx]['children'][char] = new_node_idx
                trie_nodes.append({'children': {}, 'word_indices': [], 'depth': char_idx + 1})
            curr_node_idx = trie_nodes[curr_node_idx]['children'][char]
            trie_nodes[curr_node_idx]['word_indices'].append(i)
    
    # Calculate D_k+1 and collect nodes for count k
    max_depth_k_plus_1 = 0
    nodes_for_count_k = [] 

    for node_data in trie_nodes: # Iterate using node_data directly
        count = len(node_data['word_indices'])
        depth = node_data['depth']
        
        if count >= k + 1:
            max_depth_k_plus_1 = max(max_depth_k_plus_1, depth)
        elif count == k:
            # Sort word_indices for consistent interval processing.
            # This list is W(u) for the current node.
            # Sorting is N_k * (k log k), bounded by S_total_len * log k.
            node_data['word_indices'].sort()
            nodes_for_count_k.append({'depth': depth, 'W_u': node_data['word_indices']})

    for i in range(n):
        ans[i] = max_depth_k_plus_1

    # Segment Tree for f_values
    f_values = [0] * n 
    
    if nodes_for_count_k:
        # Determine segment tree size (smallest power of 2 >= n)
        st_size = 1
        while st_size < n:
            st_size *= 2
        
        # tree_val stores actual max in range, tree_lazy stores value to propagate
        st_val = [0] * (2 * st_size)
        st_lazy = [0] * (2 * st_size)

        # Applies `val_to_apply` to node `idx` (updates its value and lazy tag)
        def apply_to_node(idx: int, val_to_apply: int):
            st_val[idx] = max(st_val[idx], val_to_apply)
            if idx < st_size: # If not a leaf, it has children, so update lazy tag
                st_lazy[idx] = max(st_lazy[idx], val_to_apply)

        # Pushes lazy tag from node `idx` to its children
        def push_lazy(idx: int):
            if st_lazy[idx] != 0 and idx < st_size: # If has lazy tag and is not leaf
                apply_to_node(2 * idx, st_lazy[idx])
                apply_to_node(2 * idx + 1, st_lazy[idx])
                st_lazy[idx] = 0 # Lazy tag cleared after propagation

        # Propagates lazy tags from parents of `idx` down to `idx` itself
        # Used before accessing/updating `idx` or its children.
        # `idx` here is for array [0, 2*st_size-1]
        def push_path(idx: int):
            # Iterate from an ancestor of idx down to parent of idx
            # Height of tree is log2(st_size). Path has this many internal nodes.
            for shift in range(st_size.bit_length() - 2, -1, -1): # height-1 down to 0
                 push_lazy(idx >> shift)
        
        # Updates range [query_L, query_R-1]
        def update_range(query_L: int, query_R: int, val: int):
            if query_L >= query_R: return # Empty range

            # Convert to 1-indexed segment tree leaves for easier path logic
            l_idx = query_L + st_size
            r_idx = query_R + st_size # exclusive endpoint

            # Push lazy tags along paths to ensure consistency before updates
            # This iterative segment tree structure often pushes "just in time" or specifically on paths.
            # For this "max" type update, we can simplify if strict segment tree properties not needed until end.
            # Standard iterative:
            # Push from roots to leaves for interval endpoints
            # This ensures values used for recalculating parents are correct.
            # Not strictly needed for this problem's lazy propagation ("max with value").
            # Simplified: directly update affected nodes.
            
            while l_idx < r_idx:
                if l_idx % 2 == 1: # l_idx is a right child
                    apply_to_node(l_idx, val)
                    l_idx += 1
                if r_idx % 2 == 1: # r_idx means original range included up to r_idx-1
                    r_idx -= 1
                    apply_to_node(r_idx, val)
                l_idx //= 2
                r_idx //= 2
        
        for item in nodes_for_count_k:
            depth_val = item['depth']
            W_u = item['W_u'] # Sorted list of k word indices

            current_pos = 0
            for forbidden_idx in W_u:
                if forbidden_idx > current_pos: # If there's a valid interval before forbidden_idx
                    update_range(current_pos, forbidden_idx, depth_val)
                current_pos = forbidden_idx + 1 
            
            if current_pos < n: # If there's a valid interval after the last forbidden_idx
                update_range(current_pos, n, depth_val)
        
        # Extract final values by pushing all lazy tags down to leaves
        for i_node in range(1, st_size): # Iterate internal nodes (from root=1 down)
            push_lazy(i_node) # Push its lazy tag to its children
        
        for i_array_idx in range(n): # Read values from leaf nodes
            f_values[i_array_idx] = st_val[st_size + i_array_idx]

    # Combine results
    for i in range(n):
        ans[i] = max(ans[i], f_values[i])
            
    return ans