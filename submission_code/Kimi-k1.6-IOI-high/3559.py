class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        # Build the trie using a dictionary of dictionaries
        root = {}
        for word in words:
            node = root
            for c in word:
                if c not in node:
                    node[c] = {}
                node = node[c]
        
        n = len(target)
        possible_lengths = []
        for i in range(n):
            lengths = []
            node = root
            l = 0
            for j in range(i, n):
                c = target[j]
                if c not in node:
                    break
                node = node[c]
                l += 1
                lengths.append(l)
            possible_lengths.append(lengths)
        
        # Initialize DP array
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(n):
            if dp[i] == float('inf'):
                continue
            for l in possible_lengths[i]:
                if i + l <= n and dp[i] + 1 < dp[i + l]:
                    dp[i + l] = dp[i] + 1
        
        return dp[n] if dp[n] != float('inf') else -1