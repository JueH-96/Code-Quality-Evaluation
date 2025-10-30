from typing import List

class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        # Build the trie
        root = {}
        for word in words:
            current = root
            for c in word:
                if c not in current:
                    current[c] = {}
                current = current[c]
        
        n = len(target)
        possible_ls = [[] for _ in range(n)]  # possible_ls[j] contains list of possible l's starting at j
        
        # Precompute possible l's for each position j in target
        for j in range(n):
            current = root
            step = 0
            while True:
                current_pos = j + step
                if current_pos >= n:
                    break
                c = target[current_pos]
                if c not in current:
                    break
                current = current[c]
                step += 1
                possible_ls[j].append(step)
        
        # Dynamic programming
        INF = float('inf')
        dp = [INF] * (n + 1)
        dp[0] = 0
        
        for i in range(n + 1):
            if dp[i] == INF:
                continue
            if i >= n:
                continue
            # Process possible_ls[i]
            for l in possible_ls[i]:
                if i + l <= n:
                    if dp[i + l] > dp[i] + 1:
                        dp[i + l] = dp[i] + 1
        
        return dp[n] if dp[n] != INF else -1