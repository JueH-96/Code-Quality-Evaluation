from typing import List

class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        # Build the trie of all prefixes
        trie = {}
        for word in words:
            current = trie
            for char in word:
                if char not in current:
                    current[char] = {}
                current = current[char]
        
        n = len(target)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # Base case: 0 characters require 0 strings
        
        for i in range(n):
            if dp[i] == float('inf'):
                continue
            
            # Find the maximum possible L for the current position i
            current = trie
            max_L = 0
            for j in range(i, n):
                char = target[j]
                if char not in current:
                    break
                current = current[char]
                max_L += 1
            
            # Update dp for all possible L from 1 to max_L
            for L in range(1, max_L + 1):
                next_pos = i + L
                if next_pos > n:
                    break
                if dp[next_pos] > dp[i] + 1:
                    dp[next_pos] = dp[i] + 1
        
        return dp[n] if dp[n] != float('inf') else -1