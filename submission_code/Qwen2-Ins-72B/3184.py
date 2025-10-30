from sortedcontainers import SortedList

class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        dp = {}  # Dictionary to store the maximum sum for a given difference
        max_sum = float('-inf')  # Initialize max_sum with negative infinity
        
        for i in range(n):
            idx = SortedList(dp.keys()).bisect_right(i - nums[i])  # Find the rightmost index with a smaller or equal difference
            if idx > 0:
                prev_diff = SortedList(dp.keys())[idx - 1]  # Get the previous difference
                dp[i - nums[i]] = max(dp.get(i - nums[i], float('-inf')), dp[prev_diff] + nums[i])  # Update the maximum sum for the current difference
            else:
                dp[i - nums[i]] = max(dp.get(i - nums[i], float('-inf')), nums[i])  # Update the maximum sum for the current difference
            
            max_sum = max(max_sum, dp[i - nums[i]])  # Update max_sum if necessary
        
        return max_sum