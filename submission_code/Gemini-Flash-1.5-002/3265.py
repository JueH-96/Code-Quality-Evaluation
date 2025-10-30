class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        max_sum = 0
        n = len(nums)
        for i in range(n):
            current_sum = 0
            for j in range(i, n):
                current_sum += nums[j]
                if abs(nums[i] - nums[j]) == k:
                    max_sum = max(max_sum, current_sum)
        return max_sum