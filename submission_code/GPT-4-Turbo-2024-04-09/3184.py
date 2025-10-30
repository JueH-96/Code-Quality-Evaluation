class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return nums[0]
        
        # dp[i] will store the maximum sum of a balanced subsequence ending at index i
        dp = [0] * n
        dp[0] = nums[0]
        
        # To track the maximum sum of any balanced subsequence found so far
        max_sum = nums[0]
        
        # Process each element in the array
        for i in range(1, n):
            dp[i] = nums[i]  # At minimum, the subsequence can just be the element itself
            # Check all previous elements to see if we can extend a balanced subsequence to include nums[i]
            for j in range(i - 1, -1, -1):
                if nums[i] - nums[j] >= i - j:
                    dp[i] = max(dp[i], dp[j] + nums[i])
            # Update the global maximum sum
            max_sum = max(max_sum, dp[i])
        
        return max_sum