import math

class Solution:
    def changes_to_palindrome(self, s: str) -> int:
        n = len(s)
        changes = 0
        for i in range(n // 2):
            if s[i] != s[n - 1 - i]:
                changes += 1
        return changes
    
    def min_changes_to_semi_palindrome(self, s: str) -> int:
        n = len(s)
        if n <= 1:
            return float('inf')
        min_changes = float('inf')
        
        # consider d=1
        changes_d1 = self.changes_to_palindrome(s)
        min_changes = min(min_changes, changes_d1)
        
        divisors = []
        for d in range(2, n):
            if n % d == 0:
                divisors.append(d)
                
        for d in divisors:
            current_changes = 0
            for i in range(d):
                substring_indices = []
                for j in range(n // d):
                    substring_indices.append(s[i + j * d])
                substring_t = "".join(substring_indices)
                current_changes += self.changes_to_palindrome(substring_t)
            min_changes = min(min_changes, current_changes)
            
        return min_changes if min_changes != float('inf') else float('inf')

    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        dp = [[float('inf')] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        for j in range(1, k + 1):
            for i in range(1, n + 1):
                for l in range(1, i + 1):
                    substring = s[i-l:i]
                    cost_substring = self.min_changes_to_semi_palindrome(substring)
                    if dp[i-l][j-1] != float('inf') and cost_substring != float('inf'):
                        dp[i][j] = min(dp[i][j], dp[i-l][j-1] + cost_substring)
                        
        result = dp[n][k]
        return result if result != float('inf') else -1 # Should not return -1 based on problem description