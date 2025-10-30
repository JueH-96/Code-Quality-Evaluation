class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        INF = 10**9
        
        def min_changes_to_palindrome(chars):
            length = len(chars)
            changes = 0
            for i in range(length // 2):
                if chars[i] != chars[length - 1 - i]:
                    changes += 1
            return changes
        
        def min_changes_to_semi_palindrome(substring):
            length = len(substring)
            if length == 1:
                return INF  # Single character cannot be semi-palindrome
            
            min_changes = INF
            
            # Try all possible values of d
            for d in range(1, length):
                if length % d == 0:
                    changes = 0
                    # For each group of indices with same modulo d
                    for mod in range(d):
                        # Get characters at indices with this modulo
                        chars = []
                        for i in range(mod, length, d):
                            chars.append(substring[i])
                        
                        # Calculate changes needed to make this group a palindrome
                        changes += min_changes_to_palindrome(chars)
                    
                    min_changes = min(min_changes, changes)
            
            return min_changes if min_changes != INF else INF
        
        # Precompute costs for all substrings
        cost = [[INF] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                cost[i][j] = min_changes_to_semi_palindrome(s[i:j+1])
        
        # DP: dp[i][j] = minimum changes to partition s[0:i] into j parts
        dp = [[INF] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, min(i, k) + 1):
                for l in range(j - 1, i):
                    if dp[l][j - 1] != INF and cost[l][i - 1] != INF:
                        dp[i][j] = min(dp[i][j], dp[l][j - 1] + cost[l][i - 1])
        
        return dp[n][k]