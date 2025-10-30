class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        count = 0
        while nums:
            if len(set(nums)) == len(nums):
                return count
            
            if len(nums) >= 3:
                nums = nums[3:]
            else:
                nums = []
            count += 1
        return count