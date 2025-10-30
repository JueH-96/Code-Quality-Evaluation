class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * (n + 1)  # dp[i] stores max length ending at i
        last = [0] * (n + 1)  # last[i] stores last element value ending at i
        sums = [0] * (n + 1)  # prefix sums
        
        # Calculate prefix sums
        for i in range(n):
            sums[i + 1] = sums[i] + nums[i]
            
        # For each position
        for i in range(1, n + 1):
            # Initialize with previous position
            dp[i] = dp[i-1]
            last[i] = last[i-1]
            
            # Find j where sum from j to i can be used
            j = i - 1
            while j >= 0:
                curr_sum = sums[i] - sums[j]
                if curr_sum >= last[j]:
                    if dp[j] + 1 > dp[i]:
                        dp[i] = dp[j] + 1
                        last[i] = curr_sum
                    break
                j -= 1
                
        return dp[n]