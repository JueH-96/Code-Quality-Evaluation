class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 2:
            return n
        
        ans = 0
        for start_parity in [0, 1]:
            current_length = 0
            for i in range(n - 1):
                if (nums[i] + nums[i+1]) % 2 == start_parity:
                    current_length += 1
            
            
            dp = [0] * n
            
            for i in range(1, n):
                if (nums[i-1] + nums[i]) % 2 == start_parity:
                    dp[i] = dp[i-1] + 1
                
            
            ans = max(ans, max(dp) + 1 if any(dp) else 0)
            
            
        
        return max(2, ans)