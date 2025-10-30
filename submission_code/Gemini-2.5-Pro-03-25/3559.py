from typing import List
import collections # Import not strictly necessary for this implementation but good practice for potential extensions.

# Definition for a Trie node.
# Each node represents a character in a prefix.
class TrieNode:
    def __init__(self):
        # children maps a character to the next TrieNode in the path.
        self.children = {}
        # is_valid_prefix_end is True if the path from the root to this node
        # forms a prefix of at least one word in the input 'words' list.
        self.is_valid_prefix_end = False

class Solution:
    """
    Solves the problem of finding the minimum number of valid strings
    that can be concatenated to form the target string.
    A string is valid if it is a prefix of any string in the words list.
    Uses Dynamic Programming with a Trie for efficient prefix checking.
    """
    def minValidStrings(self, words: List[str], target: str) -> int:
        """
        Calculates the minimum number of valid strings to form the target.

        Args:
          words: A list of strings whose prefixes are considered valid.
          target: The target string to form by concatenating valid strings.

        Returns:
          The minimum number of valid strings required, or -1 if it's impossible
          to form the target string.
        """

        # Step 1: Build the Trie containing all prefixes of the words.
        # The Trie allows efficient checking of whether a substring of 'target'
        # is a valid prefix.
        root = TrieNode()
        for word in words:
            node = root
            for char in word:
                # Traverse the Trie, creating nodes if they don't exist.
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
                # Mark every node encountered along the path of a word as
                # representing a valid prefix end. This ensures that all prefixes
                # (e.g., "a", "ab", "abc" for word "abc") are marked.
                node.is_valid_prefix_end = True
                
        n = len(target)
        
        # Step 2: Initialize DP array.
        # dp[i] will store the minimum number of valid strings needed to form
        # the prefix target[:i].
        # Initialize all dp values to infinity, signifying that states are initially unreachable.
        dp = [float('inf')] * (n + 1)
        
        # Base case: An empty prefix (target[:0]) requires 0 valid strings.
        dp[0] = 0 
        
        # Step 3: Fill the DP table using a forward approach.
        # Iterate through each possible starting position 'i' in the target string.
        # dp[i] represents the minimum cost to reach index 'i'.
        for i in range(n):
            # If dp[i] is infinity, it means the prefix target[:i] cannot be formed.
            # Thus, we cannot start a new valid string segment from index 'i'.
            if dp[i] == float('inf'):
                continue
                
            # Try to find valid prefixes starting from index 'i' of the target string.
            # Start a Trie traversal from the root for the substring target[i:].
            node = root
            # Iterate through possible end positions 'j' for a valid string segment
            # starting at index 'i'. The segment is target[i : j+1].
            for j in range(i, n):
                char = target[j]
                
                # Check if the character target[j] can extend the current path in the Trie.
                if char not in node.children:
                    # If the character is not found in the children of the current Trie node,
                    # it means that target[i : j+1] (and any longer string starting at i)
                    # cannot be a prefix found in the 'words' list. Stop extending from 'i'.
                    break 
                    
                # Move to the next node in the Trie corresponding to the character.
                node = node.children[char]
                
                # Check if the path traced so far (representing substring target[i : j+1])
                # ends at a node marked as a valid prefix end.
                if node.is_valid_prefix_end:
                    # If target[i : j+1] is a valid prefix, it means we have found a way
                    # to cover the target string up to index j+1.
                    # The cost to reach index j+1 is potentially dp[i] (cost to reach index i)
                    # plus 1 (for the current valid string segment target[i : j+1]).
                    # Update dp[j+1] with the minimum cost found so far.
                    dp[j + 1] = min(dp[j + 1], dp[i] + 1)
                    
        # Step 4: Determine the final result.
        # dp[n] contains the minimum number of valid strings to form the entire target string.
        result = dp[n]
        
        # If dp[n] is still infinity, it means the target string cannot be formed.
        if result == float('inf'):
            return -1
        else:
            # Otherwise, return the minimum count found. Cast to int as required.
            return int(result)