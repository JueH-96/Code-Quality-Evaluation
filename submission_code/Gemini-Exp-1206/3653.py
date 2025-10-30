class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_sum = float('-inf')

        for i in range(n):
            current_sum = 0
            for j in range(i, n):
                current_sum += nums[j]
                if (j - i + 1) % k == 0:
                    max_sum = max(max_sum, current_sum)

        return max_sum