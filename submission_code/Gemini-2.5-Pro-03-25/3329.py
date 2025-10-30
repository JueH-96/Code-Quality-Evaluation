from typing import List 

# Define TrieNode structure outside the Solution class for clarity.
# Each node represents a prefix character, and its children represent the next possible characters.
class TrieNode:
    def __init__(self):
        # Using a dictionary to store child nodes. 
        # Key: character ('0'-'9'), Value: TrieNode representing the next character in the prefix.
        self.children = {} 

class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        """
        Finds the length of the longest common prefix between any pair of numbers (x, y)
        such that x belongs to arr1 and y belongs to arr2.

        The approach uses a Trie (prefix tree) to efficiently store all prefixes 
        of numbers in arr1 and then searches for the longest matching prefix for 
        each number in arr2.

        Args:
            arr1: A list of positive integers.
            arr2: A list of positive integers.

        Returns:
            The length of the longest common prefix found among all pairs. 
            Returns 0 if no common prefix exists.
        """
        
        # Initialize the root node of the Trie.
        trie_root = TrieNode()
        
        # --- Step 1: Build the Trie ---
        # Populate the Trie with all prefixes derived from the numbers in arr1.
        # Each path from the root down represents a prefix present in at least one number in arr1.
        for num in arr1:
            # Convert the integer to its string representation to access digits.
            s_num = str(num)
            # Start traversal from the root for each number.
            node = trie_root
            # Iterate through each digit (character) of the number's string form.
            for digit_char in s_num:
                # Use the dictionary's setdefault method for concise node creation/retrieval:
                # - If 'digit_char' is already a child, it returns the existing child TrieNode.
                # - If 'digit_char' is not a child, it creates a new TrieNode, adds it as a child
                #   with 'digit_char' as the key, and returns the new node.
                # Update 'node' to point to the child node corresponding to the current digit.
                node = node.children.setdefault(digit_char, TrieNode())
                
        # --- Step 2: Search for the Longest Common Prefix ---
        # Initialize a variable to keep track of the maximum length found across all comparisons.
        max_prefix_len = 0
        
        # Iterate through each number in arr2 to compare its prefixes against those stored in the Trie (from arr1).
        for num in arr2:
            # Convert the integer to its string representation.
            s_num = str(num)
            # Start traversal from the Trie root for each number in arr2.
            node = trie_root
            # Keep track of the length of the common prefix found for the current number from arr2.
            current_prefix_len = 0
            # Iterate through each digit (character) of the current number from arr2.
            for digit_char in s_num:
                # Check if the current digit exists as a child of the current Trie node.
                # This indicates that the current prefix of the number from arr2 matches a prefix from arr1.
                if digit_char in node.children:
                    # If the path exists, move down to the corresponding child node.
                    node = node.children[digit_char]
                    # Increment the length of the common prefix found so far for this number.
                    current_prefix_len += 1
                    # Update the overall maximum length found. We check at each step because 
                    # the longest common prefix could end at any point.
                    max_prefix_len = max(max_prefix_len, current_prefix_len)
                else:
                    # If the digit character is not found as a child, the common prefix path ends here.
                    # No need to check further digits for this number from arr2.
                    # Break the inner loop and proceed to the next number in arr2.
                    break 
            # After checking all digits of a number from arr2 or breaking early, 
            # continue to the next number in arr2.
                    
        # --- Step 3: Return the result ---
        # After iterating through all numbers in arr2 and comparing their prefixes,
        # return the overall maximum common prefix length found.
        # If no common prefixes were found, max_prefix_len will remain 0.
        return max_prefix_len