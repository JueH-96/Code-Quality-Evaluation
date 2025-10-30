from typing import List

class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        # Build the trie structure containing all valid prefixes
        trie = {}
        for word in words:
            node = trie
            for c in word:
                if c not in node:
                    node[c] = {}
                node = node[c]
        
        n = len(target)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # Base case: zero characters require zero steps
        
        for j in range(n):
            if dp[j] == float('inf'):
                continue  # Skip positions that are unreachable
            
            current_node = trie
            max_l = 0
            # Determine the maximum valid prefix length starting at j
            for i in range(j, n):
                c = target[i]
                if c not in current_node:
                    break
                current_node = current_node[c]
                max_l += 1
            
            # Update dp[j + l] for all valid l from 1 to max_l
            for l in range(1, max_l + 1):
                if j + l > n:
                    break
                if dp[j] + 1 < dp[j + l]:
                    dp[j + l] = dp[j] + 1
        
        return dp[n] if dp[n] != float('inf') else -1