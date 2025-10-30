from typing import List

class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        # Define the TrieNode structure
        class TrieNode:
            __slots__ = ['children']
            def __init__(self):
                self.children = {}
        
        root = TrieNode()
        # Build the trie with all prefixes of all words
        for word in words:
            node = root
            for c in word:
                if c not in node.children:
                    node.children[c] = TrieNode()
                node = node.children[c]
        
        n = len(target)
        # Precompute L_max for each position i in target
        L_max = [0] * n
        for i in range(n):
            node = root
            current_length = 0
            for j in range(i, n):
                c = target[j]
                if c in node.children:
                    node = node.children[c]
                    current_length += 1
                    L_max[i] = current_length
                else:
                    break
        
        # Dynamic programming to find the minimum number of valid strings
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        for i in range(n + 1):
            if dp[i] == float('inf'):
                continue
            if i < n:
                max_jump = L_max[i]
                for jump in range(1, max_jump + 1):
                    next_pos = i + jump
                    if next_pos <= n and dp[next_pos] > dp[i] + 1:
                        dp[next_pos] = dp[i] + 1
        
        return dp[n] if dp[n] != float('inf') else -1