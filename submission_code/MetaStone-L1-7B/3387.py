class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        sorted_nums = sorted(nums)
        m = len(sorted_nums) // 2
        sum_ops = 0
        
        sum_ops += abs(k - sorted_nums[m])
        
        for i in range(m + 1, len(sorted_nums)):
            sum_ops += abs(k - sorted_nums[i])
        
        for i in range(m):
            sum_ops += abs(k - sorted_nums[i])
        
        return sum_ops