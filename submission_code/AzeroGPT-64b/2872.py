class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        def operation(nums: List[int]) -> int:
            for i in range(len(nums) - 1):
                if nums[i] <= nums[i + 1]:
                    nums[i + 1] += nums[i]
                    nums.pop(i)
                    return True
            return False
        
        while operation(nums):
            pass
        
        return nums[0]