from typing import List

class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        # Build a trie of all prefixes from the words.
        trie = {}
        for word in words:
            node = trie
            for char in word:
                # If char not in current node, create new branch.
                if char not in node:
                    node[char] = {}
                node = node[char]
        
        n = len(target)
        # dp[i] will hold the minimum number of valid strings (prefixes) needed to form target[:i]
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # No segments needed for an empty prefix

        # For each starting index, try to extend with a valid prefix from the trie.
        for i in range(n):
            if dp[i] == float('inf'):
                continue  # skip unreachable indices
            node = trie
            # Try all possible valid segments starting at i
            for j in range(i, n):
                ch = target[j]
                if ch not in node:
                    break  # no further valid prefixes from this starting position
                node = node[ch]
                # If we reach here, target[i:j+1] is a valid prefix of one of the words.
                dp[j+1] = min(dp[j+1], dp[i] + 1)
        
        return dp[n] if dp[n] != float('inf') else -1