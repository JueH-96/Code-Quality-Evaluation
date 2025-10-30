from typing import List

class Solution:
  def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
    # Using list representation for Trie node: [best_len, best_idx, children_dict]
    # Define constants for list indices for clarity.
    BEST_LEN = 0 
    BEST_IDX = 1
    CHILDREN = 2

    trie_root = [float('inf'), -1, {}] # Node structure: [length, index, children_dictionary]

    for i, wc_str in enumerate(wordsContainer):
      l_i = len(wc_str)
      
      # Update root node's information.
      # The root represents an empty suffix, so its info reflects the overall best choice among all words.
      current_node_list_repr = trie_root
      if l_i < current_node_list_repr[BEST_LEN]:
        current_node_list_repr[BEST_LEN] = l_i
        current_node_list_repr[BEST_IDX] = i
      # If l_i == current_node_list_repr[BEST_LEN]:
      #   The current current_node_list_repr[BEST_IDX] was set by an earlier word (index k < i).
      #   Criterion 3 ("occurred earlier") prefers index k. So, no change needed.
      
      # Traverse/build trie for characters of reversed wc_str
      for char_idx in range(len(wc_str) - 1, -1, -1): # Iterate from last char to first
        char = wc_str[char_idx]
        
        children_dict = current_node_list_repr[CHILDREN]
        if char not in children_dict:
          # Create a new node if this path doesn't exist
          children_dict[char] = [float('inf'), -1, {}] 
        
        current_node_list_repr = children_dict[char] # Move to the child node
        
        # Update this path node's information with (l_i, i)
        if l_i < current_node_list_repr[BEST_LEN]:
          current_node_list_repr[BEST_LEN] = l_i
          current_node_list_repr[BEST_IDX] = i
        # Similarly, if l_i == current_node_list_repr[BEST_LEN], existing BEST_IDX is preferred.
    
    ans_array = []
    for query_str in wordsQuery:
      current_node_list_repr = trie_root
      # Default answer is based on info at root (empty suffix)
      # This will be correct if query_str is empty or no non-empty common suffix is found.
      # Note: wordsContainer is guaranteed non-empty, so trie_root[BEST_IDX] will be a valid index.
      ans_for_this_query = current_node_list_repr[BEST_IDX] 
      
      # Traverse Trie with reversed query_str
      for char_idx in range(len(query_str) - 1, -1, -1): # Iterate from last char to first
        char = query_str[char_idx]
        
        children_dict = current_node_list_repr[CHILDREN]
        if char not in children_dict:
          # Path ends; current char from query_str not in trie.
          # Longest common suffix found so far is represented by previous node.
          break 
        
        # Move to child node (represents a longer common suffix)
        current_node_list_repr = children_dict[char]
        # Update answer with the best choice for this new, longer common suffix
        ans_for_this_query = current_node_list_repr[BEST_IDX]
      
      ans_array.append(ans_for_this_query)
      
    return ans_array