from typing import List
import sys

class TrieNode:
    def __init__(self):
        self.children = {}
        # min_cost stores the minimum cost of a word that ends at this node.
        # Initialize with infinity to indicate no word ends here yet.
        self.min_cost = float('inf')

class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        n = len(target)
        
        # Build Trie from the dictionary words.
        # Each node in the Trie represents a prefix of one or more words.
        # Nodes that correspond to the end of a word store the minimum cost
        # of that word (or words, if multiple words end at the same node).
        root = TrieNode()
        for word, cost in zip(words, costs):
            curr = root
            for char in word:
                # Create child node if it doesn't exist
                if char not in curr.children:
                    curr.children[char] = TrieNode()
                # Move to the next node
                curr = curr.children[char]
            # Update the minimum cost at the node where the word ends
            curr.min_cost = min(curr.min_cost, cost)

        # Dynamic Programming approach.
        # dp[i] will store the minimum cost to form the prefix of 'target'
        # of length 'i', i.e., target[0...i-1].
        # The dp table size is n + 1, to accommodate prefix lengths from 0 to n.
        # dp[0] represents the cost to form an empty string, which is 0.
        # All other dp values are initialized to infinity, indicating that the
        # corresponding prefixes are initially unreachable.
        dp = [float('inf')] * (n + 1)
        dp[0] = 0

        # Iterate through the target string. The index 'i' represents the
        # starting position in the target string from where we consider
        # appending a dictionary word. This also corresponds to the end
        # index of the prefix target[0...i-1] that we assume has already
        # been formed with cost dp[i].
        for i in range(n):
            # If the prefix target[0...i-1] is unreachable (its minimum cost
            # dp[i] is infinity), then we cannot extend this prefix by appending
            # any word. We can safely skip this starting position 'i' as it
            # cannot lead to a valid, finite-cost composition of a longer prefix.
            # This pruning is crucial for performance.
            if dp[i] == float('inf'):
                continue

            # Start traversing the Trie from the root. We are looking for dictionary
            # words that match a substring of 'target' starting at index 'i'.
            curr = root
            
            # Iterate through the target string from index 'i' onwards.
            # The index 'j' represents the current end index of the substring
            # target[i...j] that we are trying to match with a dictionary word
            # by traversing the Trie.
            for j in range(i, n):
                char = target[j]
                
                # Follow the edge in the Trie corresponding to the character target[j].
                # If there is no edge for this character from the current Trie node,
                # it means the substring target[i...j] is not a prefix of any word
                # in our dictionary. Consequently, any longer substring starting at 'i'
                # and extending beyond 'j' also cannot be a prefix of a dictionary word
                # using this path. We break the inner loop as we cannot form any more
                # dictionary words starting at index 'i' by extending the current matched prefix.
                if char not in curr.children:
                    break 
                
                # Move to the next Trie node. 'curr' now represents the node corresponding
                # to the prefix of dictionary words that matches the substring target[i...j].
                curr = curr.children[char]
                
                # Check if the current Trie node marks the end of one or more dictionary words.
                # The `min_cost` attribute at this node stores the minimum cost among all
                # dictionary words whose path from the root ends at this node.
                if curr.min_cost != float('inf'):
                    # We have found a dictionary word that exactly matches the substring
                    # target[i...j] with minimum cost `curr.min_cost`.
                    # This word allows us to transition from the state where we have successfully
                    # formed the prefix target[0...i-1] (with minimum cost dp[i]) to the state
                    # where we have formed the prefix target[0...j]. The length of this new prefix
                    # is j - 0 + 1 = j + 1. The corresponding DP state is dp[j+1].
                    
                    # Update the minimum cost to reach state j+1.
                    # The new candidate cost is the sum of the minimum cost to reach state i (dp[i])
                    # and the minimum cost of the dictionary word target[i...j] (curr.min_cost).
                    dp[j + 1] = min(dp[j + 1], dp[i] + curr.min_cost)

        # After iterating through all possible starting positions 'i' in the target
        # string and exploring all possible dictionary words that can start at 'i',
        # dp[n] will hold the minimum cost to form the entire target string
        # target[0...n-1].
        # If dp[n] is still infinity, it means the target string cannot be formed
        # by concatenating words from the dictionary.
        result = dp[n]
        return result if result != float('inf') else -1