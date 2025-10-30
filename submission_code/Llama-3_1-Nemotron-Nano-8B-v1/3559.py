from typing import List

class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        # Build the trie of all prefixes
        trie = {}
        max_len = 0
        for word in words:
            current = trie
            for c in word:
                if c not in current:
                    current[c] = {}
                current = current[c]
                # Mark this node as valid (part of a prefix)
                current['valid'] = True
                max_len = max(max_len, len(word))
        
        n = len(target)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # Base case: 0 characters need 0 valid strings
        
        for i in range(n):
            if dp[i] == float('inf'):
                continue  # Skip if current position is unreachable
            current_node = trie
            # Explore up to max_len characters ahead
            for j in range(i, min(i + max_len, n)):
                char = target[j]
                if char in current_node:
                    current_node = current_node[char]
                    # Update the DP for the next position if current node is valid
                    next_pos = j + 1
                    if dp[next_pos] > dp[i] + 1:
                        dp[next_pos] = dp[i] + 1
                else:
                    break  # No further prefixes possible from this path
        
        return dp[n] if dp[n] != float('inf') else -1