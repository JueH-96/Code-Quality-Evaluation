class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Find the break point where nums[i] > nums[i+1]
        break_count = 0
        break_index = -1
        
        for i in range(n):
            if nums[i] > nums[(i + 1) % n]:
                break_count += 1
                break_index = i
        
        # If no break point, array is already sorted
        if break_count == 0:
            return 0
        
        # If more than one break point, impossible to sort
        if break_count > 1:
            return -1
        
        # Check if rotating at break_index would result in sorted array
        # The array should be sorted from break_index+1 to end, and from start to break_index
        # Also, the last element should be <= first element after rotation
        if nums[n - 1] > nums[0]:
            return -1
        
        # The number of right shifts needed is n - (break_index + 1)
        return n - (break_index + 1)