class Solution:
    def subarraySum(self, nums: List[int]) -> int:
        total_sum = 0
        n = len(nums)
        for i in range(n):
            start = max(0, i - nums[i])
            subarray = nums[start:i+1]
            total_sum += sum(subarray)
        return total_sum