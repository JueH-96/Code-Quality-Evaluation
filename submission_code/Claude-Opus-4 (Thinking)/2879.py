class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        
        # Precompute the minimum changes for all substrings
        min_changes = [[0] * n for _ in range(n)]
        
        for start in range(n):
            for end in range(start + 1, n + 1):
                substring = s[start:end]
                min_changes[start][end - 1] = self.min_changes_to_semi_palindrome(substring)
        
        # DP
        # dp[i][j] = minimum changes to partition s[0:i] into j parts
        dp = [[float('inf')] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, min(i, k) + 1):
                for p in range(j - 1, i):
                    dp[i][j] = min(dp[i][j], dp[p][j - 1] + min_changes[p][i - 1])
        
        return dp[n][k]
    
    def min_changes_to_semi_palindrome(self, substring):
        n = len(substring)
        if n == 1:
            return float('inf')  # Single character is not a semi-palindrome
        
        min_changes = float('inf')
        
        for d in range(1, n):
            if n % d != 0:
                continue
            
            changes = 0
            for mod in range(d):
                # Get all indices with this modulo
                indices = []
                for i in range(mod, n, d):
                    indices.append(i)
                
                # Check if they form a palindrome
                left, right = 0, len(indices) - 1
                while left < right:
                    if substring[indices[left]] != substring[indices[right]]:
                        changes += 1
                    left += 1
                    right -= 1
            
            min_changes = min(min_changes, changes)
        
        return min_changes