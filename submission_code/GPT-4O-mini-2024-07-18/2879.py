class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        
        # Function to calculate the number of changes needed to make a substring semi-palindrome
        def changes_to_semi_palindrome(sub: str) -> int:
            length = len(sub)
            if length < 2:
                return 0
            
            # Try all possible d values
            min_changes = float('inf')
            for d in range(1, length):
                if length % d == 0:
                    changes = 0
                    # Check each group of indices
                    for i in range(d):
                        count = {}
                        for j in range(i, length, d):
                            count[sub[j]] = count.get(sub[j], 0) + 1
                        max_freq = max(count.values(), default=0)
                        changes += (length // d) - max_freq
                    min_changes = min(min_changes, changes)
            return min_changes
        
        # Dynamic programming to find the minimum changes for k partitions
        dp = [[float('inf')] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, min(k, i) + 1):
                for p in range(j - 1, i):
                    dp[i][j] = min(dp[i][j], dp[p][j - 1] + changes_to_semi_palindrome(s[p:i]))
        
        return dp[n][k]