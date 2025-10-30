class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        mod = 10 ** 9 + 7
        
        # Dynamic programming to calculate the maximum sum of a subsequence with non-adjacent elements
        dp = [0] * (n + 1)
        dp[0] = 0
        dp[1] = nums[0]
        for i in range(2, n + 1):
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i - 1])
        
        # Process the queries
        result = 0
        for pos, x in queries:
            # Update the value at the given position
            old_value = nums[pos]
            nums[pos] = x
            
            # Recalculate the maximum sum of a subsequence with non-adjacent elements
            new_dp = [0] * (n + 1)
            new_dp[0] = 0
            new_dp[1] = nums[0]
            for i in range(2, n + 1):
                new_dp[i] = max(new_dp[i - 1], new_dp[i - 2] + nums[i - 1])
            
            # Add the answer to the result
            result = (result + new_dp[n]) % mod
            
            # Restore the original value
            nums[pos] = old_value
        
        return result