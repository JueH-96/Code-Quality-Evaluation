class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_sum = -float('inf')

        for sub_len in range(k, n + 1, k):
            for i in range(n - sub_len + 1):
                current_sum = sum(nums[i:i + sub_len])
                max_sum = max(max_sum, current_sum)

        return max_sum