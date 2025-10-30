class Solution:
    def subarraySum(self, nums: List[int]) -> int:
        total_sum = 0
        for i in range(len(nums)):
            start = max(0, i - nums[i])
            subarray = nums[start:i+1]
            total_sum += sum(subarray)
        return total_sum