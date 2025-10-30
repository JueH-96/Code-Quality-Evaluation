class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        
        # Precompute the minimum changes needed to make s[i:j+1] a semi-palindrome
        min_changes = [[float('inf')] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i, n):
                length = j - i + 1
                if length == 1:
                    # Single character cannot be a semi-palindrome
                    min_changes[i][j] = float('inf')
                    continue
                
                # Try all possible divisors d
                for d in range(1, length):
                    if length % d == 0:
                        changes = 0
                        # Check each group of indices with same modulo d
                        for start in range(d):
                            # Get all positions with modulo d == start
                            positions = []
                            for pos in range(start, length, d):
                                positions.append(i + pos)
                            
                            # Count changes needed to make this group a palindrome
                            left = 0
                            right = len(positions) - 1
                            while left < right:
                                if s[positions[left]] != s[positions[right]]:
                                    changes += 1
                                left += 1
                                right -= 1
                        
                        min_changes[i][j] = min(min_changes[i][j], changes)
        
        # dp[i][j] = minimum changes to partition s[0:i] into j parts
        dp = [[float('inf')] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, min(i, k) + 1):
                # Try all possible last substring s[l:i]
                for l in range(j - 1, i):
                    if l == 0:
                        if j == 1 and min_changes[l][i-1] != float('inf'):
                            dp[i][j] = min(dp[i][j], min_changes[l][i-1])
                    else:
                        if dp[l][j-1] != float('inf') and min_changes[l][i-1] != float('inf'):
                            dp[i][j] = min(dp[i][j], dp[l][j-1] + min_changes[l][i-1])
        
        return dp[n][k]