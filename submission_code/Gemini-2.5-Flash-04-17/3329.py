from typing import List

class TrieNode:
    """A node in the Trie (prefix tree)."""
    def __init__(self):
        # Dictionary mapping digit character ('0'-'9') to child TrieNode
        self.children = {}

class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        """
        Finds the length of the longest common prefix between all pairs of
        integers (x, y) where x is in arr1 and y is in arr2.

        Args:
            arr1: A list of positive integers.
            arr2: A list of positive integers.

        Returns:
            The maximum length of a common prefix found among all pairs.
            Returns 0 if no common prefix exists for any pair.
        """
        # Build a Trie containing all prefixes of numbers in arr1.
        # Any node reached by traversing digits represents a prefix
        # present in at least one number in arr1.
        root = TrieNode()
        for num in arr1:
            s_num = str(num)
            curr = root
            for char in s_num:
                if char not in curr.children:
                    curr.children[char] = TrieNode()
                curr = curr.children[char]
            # After processing a number, all nodes along the path
            # from the root represent prefixes from arr1.

        # Now, iterate through arr2. For each number in arr2, traverse the
        # Trie built from arr1 as far as possible. The maximum depth reached
        # for any number in arr2 corresponds to the longest common prefix
        # length found across all pairs.
        max_lcp_len = 0
        for num in arr2:
            s_num = str(num)
            curr = root
            current_lcp_len = 0
            for char in s_num:
                if char in curr.children:
                    # If the current character exists as a child in the Trie,
                    # it means the prefix of the arr2 number up to this character
                    # is also a prefix of at least one number in arr1.
                    curr = curr.children[char]
                    current_lcp_len += 1
                    # Update the maximum length found so far.
                    max_lcp_len = max(max_lcp_len, current_lcp_len)
                else:
                    # The current character does not match a prefix in arr1's Trie.
                    # The common prefix for this specific number from arr2 ends here.
                    break # Move to the next number in arr2.

        return max_lcp_len