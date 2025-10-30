from typing import List

class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        # A Trie (prefix tree) to store all prefixes from arr1.
        # Each node in the Trie will represent a digit in a prefix.
        # We use dictionaries to represent Trie nodes, where keys are digits '0'-'9'
        # and values are child nodes (other dictionaries).
        # A special key, '_is_prefix_from_arr1', will be set to True in a node
        # if the path from the root to this node forms a valid prefix of ANY number in arr1.
        
        trie = {} # The root of the Trie
        
        # Step 1: Build the Trie by inserting all prefixes of numbers from arr1.
        for num_x in arr1:
            s_x = str(num_x)
            current_node = trie
            for i in range(len(s_x)):
                char_digit = s_x[i]
                # If the character (digit) is not a child of the current node, create a new child node.
                if char_digit not in current_node:
                    current_node[char_digit] = {}
                # Move to the child node.
                current_node = current_node[char_digit]
                # Mark this node to indicate that the string formed by the path from the root
                # to this node is a prefix of at least one number in arr1.
                current_node['_is_prefix_from_arr1'] = True 
        
        # Step 2: Traverse the Trie with numbers from arr2 to find the maximum common prefix length.
        max_lcp_length = 0
        
        for num_y in arr2:
            s_y = str(num_y)
            current_node = trie
            current_lcp = 0 # Length of the common prefix found for the current num_y
            
            for char_digit in s_y:
                # Check if the current digit exists as a child in the current Trie node.
                if char_digit in current_node:
                    # Move to the next node in the Trie.
                    current_node = current_node[char_digit]
                    current_lcp += 1 # Increment the length of the common prefix.
                    
                    # If the current path (prefix) exists in arr1 (marked during Trie construction),
                    # update the overall maximum common prefix length.
                    if current_node.get('_is_prefix_from_arr1'):
                        max_lcp_length = max(max_lcp_length, current_lcp)
                else:
                    # If the current digit is not found, it means the common prefix cannot extend further.
                    # Break out of this inner loop and proceed to the next number in arr2.
                    break
        
        return max_lcp_length