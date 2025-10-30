class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_sum = 0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(nums[i] - nums[j]) == k:
                    current_sum = 0
                    for l in range(i, j + 1):
                        current_sum += nums[l]
                    max_sum = max(max_sum, current_sum)
        return max_sum