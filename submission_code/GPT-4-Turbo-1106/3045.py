class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        # Find the index of the smallest element
        min_index = nums.index(min(nums))
        
        # Check if the array is already sorted
        if nums == sorted(nums):
            return 0
        
        # Check if the array can be sorted by right shifts
        # by comparing the sorted array with the rotated one
        for shift in range(1, len(nums)):
            if nums[min_index:] + nums[:min_index] == sorted(nums):
                return shift
            min_index = (min_index - 1) % len(nums)
        
        # If the array cannot be sorted by right shifts, return -1
        return -1