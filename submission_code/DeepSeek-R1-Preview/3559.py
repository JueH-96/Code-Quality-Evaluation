class Solution:
    def minValidStrings(self, words, target):
        # Build the trie
        trie = {}
        max_len = 0
        for word in words:
            max_len = max(max_len, len(word))
            node = trie
            for char in word:
                if char not in node:
                    node[char] = {}
                node = node[char]
        
        n = len(target)
        if n == 0:
            return 0  # Assuming target is non-empty per problem constraints, but handle anyway
        
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # Base case: 0 characters require 0 valid strings
        
        for i in range(n):
            if dp[i] == float('inf'):
                continue  # Can't reach this position
            
            node = trie
            current_l = 0
            possible_ls = []
            # Check up to max_len characters ahead
            for j in range(i, min(i + max_len, n)):
                char = target[j]
                if char in node:
                    node = node[char]
                    current_l += 1
                    possible_ls.append(current_l)
                else:
                    break  # No further characters can form a prefix
            
            for l in possible_ls:
                next_pos = i + l
                if next_pos <= n:
                    dp[next_pos] = min(dp[next_pos], dp[i] + 1)
        
        return dp[n] if dp[n] != float('inf') else -1