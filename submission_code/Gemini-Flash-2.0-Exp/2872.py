class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        n = len(nums)
        
        i = 0
        while i < len(nums) - 1:
            if nums[i] <= nums[i+1]:
                nums[i+1] = nums[i] + nums[i+1]
                nums.pop(i)
                i = 0
            else:
                i += 1
        
        return max(nums)