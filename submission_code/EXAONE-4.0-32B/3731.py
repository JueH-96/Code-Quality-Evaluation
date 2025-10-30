class Solution:
    def subarraySum(self, nums: List[int]) -> int:
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix[i] = prefix[i - 1] + nums[i - 1]
        
        total_sum = 0
        for i in range(n):
            start = max(0, i - nums[i])
            subarray_sum = prefix[i + 1] - prefix[start]
            total_sum += subarray_sum
        
        return total_sum