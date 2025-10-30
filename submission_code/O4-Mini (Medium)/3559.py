from typing import List

class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        # Build a trie of all words so that any prefix of a word is a valid string
        trie = {}
        for w in words:
            node = trie
            for ch in w:
                if ch not in node:
                    node[ch] = {}
                node = node[ch]
        
        n = len(target)
        # dp[i] = minimum pieces to form target[i:]
        # initialize with a large value; dp[n] = 0 (empty suffix needs 0 pieces)
        INF = n + 1
        dp = [INF] * (n + 1)
        dp[n] = 0
        
        # Fill dp from the end backwards
        for i in range(n - 1, -1, -1):
            node = trie
            # Try extending a prefix starting at i
            for j in range(i, n):
                ch = target[j]
                if ch not in node:
                    break
                node = node[ch]
                # If we can match target[i:j+1], update dp[i]
                dp[i] = min(dp[i], 1 + dp[j + 1])
                # Early exit if we already hit the best possible (1)
                if dp[i] == 1:
                    break
        
        return dp[0] if dp[0] < INF else -1