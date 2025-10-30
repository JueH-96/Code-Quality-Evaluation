from typing import List

class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        """
        Finds the length of the longest common prefix between all pairs of integers
        from arr1 and arr2.
        """
        
        # A Trie (prefix tree) is an ideal data structure for problems involving prefixes.
        # The overall approach is to:
        # 1. Build a Trie containing all numbers from arr1. Each path from the
        #    root of the Trie will represent a prefix of a number in arr1.
        # 2. Iterate through the numbers in arr2 and for each number, traverse
        #    the Trie to find the longest possible prefix match.
        # 3. Keep track of the maximum length found across all numbers in arr2.

        # We can implement the Trie using nested dictionaries, where each dictionary
        # represents a node and its keys are the digits leading to child nodes.
        trie = {}

        # Step 1: Build the Trie from arr1.
        for num in arr1:
            # Convert the number to its string representation.
            s = str(num)
            node = trie
            # Insert the string into the Trie character by character.
            for char in s:
                # node.setdefault(char, {}) is a concise way to:
                # 1. Get the child node for the character 'char'.
                # 2. If it doesn't exist, create a new empty dictionary for it.
                # 3. Return the child node (either existing or newly created).
                node = node.setdefault(char, {})

        # Step 2 & 3: Search for the longest common prefix and track the maximum length.
        max_len = 0
        for num in arr2:
            s = str(num)
            node = trie
            current_len = 0
            # Traverse the Trie using the digits of the current number from arr2.
            for char in s:
                if char in node:
                    # If a path exists for the current digit, follow it.
                    node = node[char]
                    current_len += 1
                else:
                    # If there's no path, the common prefix ends here.
                    break
            
            # Update the maximum length found so far.
            max_len = max(max_len, current_len)

        return max_len