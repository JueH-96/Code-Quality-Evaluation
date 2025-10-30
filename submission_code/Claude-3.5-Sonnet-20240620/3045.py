class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        pivot = 0
        
        # Find the pivot point where the array is "broken"
        for i in range(1, n):
            if nums[i] < nums[i-1]:
                pivot = i
                break
        
        # If pivot is 0, the array is already sorted
        if pivot == 0:
            return 0
        
        # Check if the array can be sorted with right shifts
        if nums[-1] > nums[0]:
            return -1
        
        # Check if the array is sorted after the pivot
        for i in range(pivot + 1, n):
            if nums[i] < nums[i-1]:
                return -1
        
        # Check if the first element is greater than the last element before pivot
        if nums[0] < nums[pivot-1]:
            return -1
        
        # Return the number of right shifts needed
        return n - pivot