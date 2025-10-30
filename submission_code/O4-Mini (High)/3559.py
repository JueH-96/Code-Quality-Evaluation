from typing import List

class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        # Build a trie of all words
        trie = {}
        for w in words:
            node = trie
            for c in w:
                if c not in node:
                    node[c] = {}
                node = node[c]
        
        n = len(target)
        INF = 10**9
        # dp[i] = minimum number of valid strings to form target[i:]
        dp = [INF] * (n + 1)
        dp[n] = 0  # empty suffix needs 0 strings
        
        # Fill dp from the end backwards
        for i in range(n - 1, -1, -1):
            node = trie
            # Try every possible prefix of target starting at i
            for j in range(i, n):
                nxt = node.get(target[j])
                if nxt is None:
                    break
                node = nxt
                # If we can form suffix from j+1, update dp[i]
                if dp[j + 1] + 1 < dp[i]:
                    dp[i] = dp[j + 1] + 1
        
        return dp[0] if dp[0] < INF else -1