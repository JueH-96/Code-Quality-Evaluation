class Solution:
    def maximumSubarraySum(self, nums, k):
        n = len(nums)
        max_sum = float('-inf')
        for i in range(n):
            for j in range(i+1, n):
                if abs(nums[i] - nums[j]) == k:
                    max_sum = max(max_sum, sum(nums[i:j+1]))
        return max_sum if max_sum != float('-inf') else 0