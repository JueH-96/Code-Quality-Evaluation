from typing import List

class TrieNode:
    def __init__(self):
        self.children = {}

class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        # Build a trie of all words to allow prefix matching
        root = TrieNode()
        for w in words:
            node = root
            for ch in w:
                if ch not in node.children:
                    node.children[ch] = TrieNode()
                node = node.children[ch]
        
        n = len(target)
        # dp[i] = minimum segments to form target[i:]
        INF = float('inf')
        dp = [INF] * (n + 1)
        dp[n] = 0  # empty suffix needs 0 segments
        
        # Fill dp from right to left
        for i in range(n - 1, -1, -1):
            node = root
            # Try to match as long a prefix as possible from position i
            for j in range(i, n):
                ch = target[j]
                if ch not in node.children:
                    break
                node = node.children[ch]
                # If we can match target[i:j+1], consider taking this prefix
                if dp[j + 1] != INF:
                    dp[i] = min(dp[i], dp[j + 1] + 1)
        
        return dp[0] if dp[0] != INF else -1