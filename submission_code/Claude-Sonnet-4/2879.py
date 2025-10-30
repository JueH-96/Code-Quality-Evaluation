class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        
        # Precompute minimum changes needed for each substring to become semi-palindrome
        def min_changes_to_semi_palindrome(start, end):
            length = end - start + 1
            if length == 1:
                return float('inf')  # Single character can't be semi-palindrome
            
            min_changes = float('inf')
            
            # Try all possible divisors d
            for d in range(1, length):
                if length % d == 0:
                    changes = 0
                    # For each remainder class modulo d
                    for r in range(d):
                        # Collect characters at positions with remainder r
                        chars = []
                        for i in range(start + r, end + 1, d):
                            chars.append(s[i])
                        
                        # Calculate changes needed to make this sequence palindrome
                        chars_len = len(chars)
                        for i in range(chars_len // 2):
                            if chars[i] != chars[chars_len - 1 - i]:
                                changes += 1
                    
                    min_changes = min(min_changes, changes)
            
            return min_changes if min_changes != float('inf') else 0
        
        # Precompute all substring costs
        cost = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                cost[i][j] = min_changes_to_semi_palindrome(i, j)
        
        # DP: dp[i][j] = minimum changes to partition first i characters into j parts
        dp = [[float('inf')] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, min(i, k) + 1):
                # Try all possible positions for the j-th partition
                for prev in range(j - 1, i):
                    if dp[prev][j - 1] != float('inf'):
                        dp[i][j] = min(dp[i][j], dp[prev][j - 1] + cost[prev][i - 1])
        
        return dp[n][k]