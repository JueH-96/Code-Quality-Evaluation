class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        if n <= 2:
            return True
        
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + nums[i]
        
        dp = [[False] * n for _ in range(n)]
        
        for i in range(n):
            dp[i][i] = True
        for i in range(n-1):
            dp[i][i+1] = True
        
        for L in range(3, n+1):
            for i in range(0, n - L + 1):
                j = i + L - 1
                found = False
                for k in range(i, j):
                    left_length = k - i + 1
                    right_length = j - k
                    left_sum = prefix[k+1] - prefix[i]
                    right_sum = prefix[j+1] - prefix[k+1]
                    
                    left_ok = (left_length == 1) or (left_sum >= m)
                    right_ok = (right_length == 1) or (right_sum >= m)
                    
                    if left_ok and right_ok and dp[i][k] and dp[k+1][j]:
                        dp[i][j] = True
                        found = True
                        break
        return dp[0][n-1]