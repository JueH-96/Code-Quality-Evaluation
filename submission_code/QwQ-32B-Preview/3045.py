class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        drop_index = -1
        
        # Find the index where the drop occurs
        for i in range(n):
            if nums[i] > nums[(i+1) % n]:
                if drop_index != -1:
                    return -1  # More than one drop
                drop_index = i
        
        # If no drops, array is already sorted
        if drop_index == -1:
            return 0
        
        # Check if both parts are sorted
        # First part: from 0 to drop_index
        # Second part: from drop_index+1 to n-1
        if not (self.is_sorted(nums, 0, drop_index) and self.is_sorted(nums, drop_index+1, n-1)):
            return -1
        
        # Check if the last element of the second part is smaller than the first element of the first part
        if nums[n-1] >= nums[0]:
            return -1
        
        # Calculate the number of right shifts needed
        return n - drop_index - 1
    
    def is_sorted(self, nums, start, end):
        for i in range(start, end):
            if nums[i] >= nums[i+1]:
                return False
        return True