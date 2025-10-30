from typing import List
import math

# Define the TrieNode class
class TrieNode:
    def __init__(self):
        # Dictionary to store children nodes, keyed by character
        self.children = {}

class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        # Build the Trie. We insert all words into the Trie.
        # A path from the root to any node represents a prefix of at least one word.
        # This is exactly what a "valid string" is defined as: a prefix of any string in words.
        root = TrieNode()
        for word in words:
            node = root
            for char in word:
                # If the current character path doesn't exist, create a new node
                if char not in node.children:
                    node.children[char] = TrieNode()
                # Move to the next node
                node = node.children[char]

        # Dynamic Programming
        n = len(target)
        # dp[i] will store the minimum number of valid strings
        # needed to form the prefix target[:i].
        # Initialize with infinity, except for dp[0].
        dp = [math.inf] * (n + 1)
        dp[0] = 0 # Base case: 0 valid strings needed to form an empty prefix target[:0]

        # Iterate through the target string using index i as the end of the current prefix target[:i]
        # More precisely, iterate through the *start* index i of potential next valid strings
        for i in range(n):
            # If the prefix target[:i] cannot be formed with a finite number of valid strings,
            # then we cannot extend from this position i. Skip this state.
            if dp[i] == math.inf:
                continue

            # Start a Trie traversal from the root.
            # We will match substrings of target starting at index i against prefixes in the Trie.
            current_trie_node = root
            # Iterate through possible lengths k of the next valid string starting at index i.
            # The substring is target[i : i+k].
            for k in range(1, n - i + 1):
                # Get the next character from the target string to match in the Trie
                char = target[i + k - 1]

                # Attempt to traverse the Trie with the current character
                if char not in current_trie_node.children:
                    # The character 'char' at target[i+k-1] does not follow the current
                    # path in the Trie (which corresponds to target[i : i+k-1]).
                    # This means the substring target[i : i+k] is not a prefix of any word.
                    # Consequently, any longer substring starting at index i (target[i : i+k'] where k' > k)
                    # will also not be a prefix. We can stop checking extensions from index i
                    # along this specific character path.
                    break

                # If the character exists in the children, move to the next Trie node.
                # The path from the root to 'current_trie_node' now represents the substring target[i : i+k].
                current_trie_node = current_trie_node.children[char]

                # Since the path for target[i : i+k] exists in the Trie, it means
                # target[i : i+k] is a prefix of at least one word in 'words'.
                # Therefore, target[i : i+k] is a valid string.

                # We can form the prefix target[:i+k] by using the minimum strings required
                # for target[:i] (which is dp[i]) and appending the current valid string
                # target[i : i+k] (which adds 1 string).
                # Update the minimum number of strings needed to reach index i+k.
                dp[i + k] = min(dp[i + k], dp[i] + 1)

        # The final answer is stored in dp[n], representing the minimum strings
        # needed to form the entire target string target[:n].
        if dp[n] == math.inf:
            return -1 # If dp[n] is still infinity, it means target cannot be formed.
        else:
            return dp[n] # Return the minimum count.