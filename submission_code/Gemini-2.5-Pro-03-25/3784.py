import sys 
# Although the implementation is iterative, setting a higher recursion depth 
# might be relevant for other system limits potentially hit with deep structures or large data.
# Usually not needed for standard iterative algorithms, but doesn't hurt.
# sys.setrecursionlimit(200000) 

from typing import List

class Solution:
  """
  Solves the Longest Common Prefix problem after removing one element.
  Finds the length of the longest common prefix (LCP) among any k strings 
  from the remaining array after removing the i-th element, for each i.
  """
  def longestCommonPrefix(self, words: List[str], k: int) -> List[int]:
    """
    Computes the longest common prefix length for k strings after removing each element.

    Args:
      words: A list of strings.
      k: The number of strings to consider for LCP.

    Returns:
      A list where answer[i] is the LCP length after removing words[i]. If removing words[i] leaves fewer than k strings, answer[i] is 0.
    """
    N = len(words)
    
    # Handle edge case k=1 separately.
    # For k=1, the LCP of 1 string is just its length. We need the maximum length among the remaining N-1 strings.
    if k == 1:
        lengths = [len(w) for w in words]
        
        # Base cases for small N
        if N == 0: return []
        if N == 1: return [0] # Removing the only element leaves an empty list. LCP length is 0.

        # Find the maximum length and the second maximum length among all words.
        max_len = 0
        max_len_indices = [] # Store indices of strings with max_len
        
        # First pass to find the maximum length
        for length in lengths:
            if length > max_len:
                max_len = length
        
        # Second pass to find all indices with max_len
        for i in range(N):
             if lengths[i] == max_len:
                 max_len_indices.append(i)

        # Determine the second maximum length considering all strings
        second_max_len = 0
        # If there are multiple strings with the maximum length
        if len(max_len_indices) > 1:
             # If we remove one max-length string, another one remains, so max length is still max_len.
             # If we remove a non-max-length string, the max length is still max_len.
             # Effectively, the second_max_len needed for the calculation when removing a unique max length string is max_len.
             second_max_len = max_len 
        else: # Exactly one string has max length. Need to find the true second max length among others.
             for i in range(N):
                 if i != max_len_indices[0]: # Check only strings other than the unique max length one
                      if lengths[i] > second_max_len:
                          second_max_len = lengths[i]

        ans = [0] * N
        for i in range(N):
            # If the removed string is the unique maximum length string
            if lengths[i] == max_len and len(max_len_indices) == 1:
                 # The new maximum length among remaining strings is the second_max_len found earlier.
                 ans[i] = second_max_len
            else: # Otherwise, the maximum length string (or one of them) is still present
                 # The maximum length among remaining strings remains max_len.
                 ans[i] = max_len
        return ans

    # General case for k > 1
    
    # If removing any element leaves fewer than k strings, answer is 0 for all.
    # The remaining array size is N-1. If N-1 < k, then we cannot choose k strings.
    if N - 1 < k:
        return [0] * N

    # Build Trie using dictionary representation
    # Store node structure (children) and node data (indices, depth) separately for potential memory efficiency and clarity.
    trie_nodes = {} # Maps node_id -> {'children': {char: child_id}}
    node_data = {} # Maps node_id -> {'indices': set(), 'depth': int}
    node_count = 0 # Counter for node IDs, start from 1 for non-root nodes
    root_id = 0
    trie_nodes[root_id] = {'children': {}}
    node_data[root_id] = {'indices': set(), 'depth': 0}

    # Populate the Trie with all words and record indices at each node.
    for idx, word in enumerate(words):
        curr_node_id = root_id
        node_data[curr_node_id]['indices'].add(idx) # Add index to root node
        
        for char_idx, char in enumerate(word):
            depth = char_idx + 1
            
            # Ensure the current node exists in trie_nodes (should always be true after root init)
            if curr_node_id not in trie_nodes:
                 trie_nodes[curr_node_id] = {'children': {}} # Safeguard

            node_children = trie_nodes[curr_node_id]['children']
            
            # If character path doesn't exist, create new node
            if char not in node_children:
                node_count += 1
                new_node_id = node_count
                node_children[char] = new_node_id
                # Initialize new node structure and data
                trie_nodes[new_node_id] = {'children': {}}
                node_data[new_node_id] = {'indices': set(), 'depth': depth}
                curr_node_id = new_node_id
            else: # Character path exists, move to existing child node
                 curr_node_id = node_children[char]
            
            # Add the word's index to the set of indices passing through this node
            node_data[curr_node_id]['indices'].add(idx)

    # Calculate L_max: The maximum depth of a node covering MORE than k strings.
    # Also collect nodes for V_k: nodes covering exactly k strings.
    L_max = 0
    V_k_nodes = [] # Stores node_ids for nodes with count k
    
    # Iterate through all nodes created (node IDs are 0 to node_count).
    for node_id in range(node_count + 1):
        # Ensure node_id is valid and has data associated (it should)
        if node_id in node_data: 
             data = node_data[node_id]
             count = len(data['indices'])
             depth = data['depth']
             
             # If count > k, this node's depth contributes to L_max calculation.
             if count > k:
                 L_max = max(L_max, depth)
             # If count == k, this node is relevant for calculating M_i values.
             elif count == k:
                 V_k_nodes.append(node_id)
    
    # Initialize the answer array. Each answer[i] must be at least L_max.
    ans = [L_max] * N

    # Process nodes in V_k to potentially update ans based on M_i calculation.
    # M_i = max depth of a node 'v' in V_k such that i is NOT in indices(v).
    if not V_k_nodes: # If V_k is empty, no further updates needed.
        return ans

    # Group nodes in V_k by their depth for efficient processing.
    depth_to_nodes = {}
    for node_id in V_k_nodes:
        depth = node_data[node_id]['depth']
        if depth not in depth_to_nodes:
            depth_to_nodes[depth] = []
        depth_to_nodes[depth].append(node_id)

    # Get distinct depths present in V_k and sort them descendingly.
    distinct_depths = sorted(depth_to_nodes.keys(), reverse=True)

    # Keep track of indices whose final M_i component hasn't been determined.
    # Initially, all indices are pending.
    pending_indices = set(range(N))
    
    # Iterate through depths in descending order. The first depth encountered for a pending index determines its M_i value.
    for depth in distinct_depths:
        # Optimization: if all indices have been processed, stop early.
        if not pending_indices: 
             break
        
        # Get all nodes in V_k that are at the current depth.
        nodes_at_depth = depth_to_nodes[depth]
        
        # Compute IndicesToExclude: the union of indices from all nodes at this depth.
        # An index i is excluded if it belongs to the index set of ANY node at this depth.
        IndicesToExclude = set()
        for node_id in nodes_at_depth:
             # Set update efficiently builds the union.
             IndicesToExclude.update(node_data[node_id]['indices'])

        # Prepare the set of indices that will remain pending for the next (lower) depth.
        next_pending_indices = set() 

        # Iterate through the currently pending indices.
        for i in pending_indices:
             # If index i is NOT excluded at this depth level...
             if i not in IndicesToExclude:
                 # ...then this 'depth' is the maximum possible depth for index i from V_k nodes.
                 # Finalize ans[i] = max(current ans[i] (which is L_max initially), this depth).
                 ans[i] = max(ans[i], depth) 
                 # This index 'i' is now fully processed, do not carry over to the next pending set.
             else:
                 # Index i is excluded at this depth, so it remains pending.
                 # Add it to the set for the next iteration (lower depth).
                 next_pending_indices.add(i)

        # Update the set of pending indices for the next iteration.
        pending_indices = next_pending_indices 
        
    # After iterating through all relevant depths, 'ans' contains the final computed values.
    return ans