class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        prefix = [0] * (n + 1)
        suffix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] | nums[i]
            suffix[n - i] = suffix[n - i + 1] | nums[n - i - 1]
        max_subarrays = 0
        for i in range(n):
            max_subarrays = max(max_subarrays, prefix[i] | suffix[i + 1])
        return max_subarrays