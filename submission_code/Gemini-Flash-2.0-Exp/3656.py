class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        for ops in range((n // 3) + 2):
            temp_nums = nums[ops * 3:]
            if len(set(temp_nums)) == len(temp_nums):
                return ops
            if len(temp_nums) == 0:
                return ops
        return (n // 3) + 1