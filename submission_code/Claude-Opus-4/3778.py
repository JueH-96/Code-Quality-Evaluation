class Solution:
    def transformArray(self, nums: List[int]) -> List[int]:
        # Replace even with 0, odd with 1
        for i in range(len(nums)):
            if nums[i] % 2 == 0:
                nums[i] = 0
            else:
                nums[i] = 1
        
        # Sort in non-decreasing order
        nums.sort()
        
        return nums