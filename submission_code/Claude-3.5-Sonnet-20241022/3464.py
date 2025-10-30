class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Calculate cost for subarray nums[l..r]
        def cost(l: int, r: int) -> int:
            result = 0
            for i in range(l, r + 1):
                if (i - l) % 2 == 0:
                    result += nums[i]
                else:
                    result -= nums[i]
            return result
        
        # dp[i] represents maximum cost achievable for subarray nums[0..i]
        dp = [0] * n
        
        # Base case - cost for first element
        dp[0] = nums[0]
        
        # Fill dp array
        for i in range(1, n):
            # Option 1: Add current element to previous subarray
            dp[i] = cost(0, i)
            
            # Option 2: Try all possible splits before current index
            for j in range(i):
                current_cost = dp[j] + cost(j + 1, i)
                dp[i] = max(dp[i], current_cost)
                
        return dp[n - 1]