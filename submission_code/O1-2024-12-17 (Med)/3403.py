class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        # Edge case
        if n == 1:
            return 1
        
        # Precompute prefix frequency counts for each character
        # prefix_counts[c][i] = number of times character c appears in s[:i]
        prefix_counts = [[0]*(n+1) for _ in range(26)]
        for i in range(n):
            for c in range(26):
                prefix_counts[c][i+1] = prefix_counts[c][i]
            prefix_counts[ord(s[i]) - ord('a')][i+1] += 1
        
        # dp[i] will hold the minimum number of balanced substrings for s[:i]
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # base case: no substrings for empty string
        
        # Helper function to check if s[j:i] is balanced
        def is_balanced(j, i):
            freqs = []
            for c in range(26):
                freq = prefix_counts[c][i] - prefix_counts[c][j]
                if freq > 0:
                    freqs.append(freq)
            if not freqs:
                return False  # empty substring doesn't make sense, but safe check
            return len(set(freqs)) == 1
        
        # Fill dp array using the definition
        for i in range(1, n + 1):
            for j in range(i):
                if dp[j] != float('inf') and is_balanced(j, i):
                    dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[n]