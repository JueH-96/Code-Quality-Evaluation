from typing import List
import sys

class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        # Define the TrieNode class
        class TrieNode:
            __slots__ = ['children', 'cost']
            def __init__(self):
                self.children = dict()
                self.cost = sys.maxsize  # Initialize with infinity
        
        # Build the trie structure
        root = TrieNode()
        for word, cost in zip(words, costs):
            current = root
            for c in word:
                if c not in current.children:
                    current.children[c] = TrieNode()
                current = current.children[c]
            # Update the cost to the minimum for this word
            if current.cost > cost:
                current.cost = cost
        
        n = len(target)
        dp = [sys.maxsize] * (n + 1)
        dp[0] = 0  # Base case: empty string has cost 0
        
        for i in range(n):
            if dp[i] == sys.maxsize:
                continue  # Skip unreachable positions
            current = root
            for j in range(i, n):
                c = target[j]
                if c not in current.children:
                    break  # No further matches in trie
                current = current.children[c]
                # If current node has a valid cost, update dp[j+1]
                if current.cost != sys.maxsize:
                    total_cost = dp[i] + current.cost
                    if total_cost < dp[j + 1]:
                        dp[j + 1] = total_cost
        
        return dp[n] if dp[n] != sys.maxsize else -1