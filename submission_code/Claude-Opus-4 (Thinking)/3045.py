class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Find all positions where the circular order breaks
        break_count = 0
        break_pos = -1
        
        for i in range(n):
            if nums[i] > nums[(i + 1) % n]:
                break_count += 1
                break_pos = i
        
        # If no breaks (shouldn't happen with distinct integers)
        if break_count == 0:
            return 0
        
        # If more than one break, can't be sorted by rotation
        if break_count > 1:
            return -1
        
        # If the only break is at the last position, already sorted
        if break_pos == n - 1:
            return 0
        
        # Otherwise, calculate right shifts needed
        return n - (break_pos + 1)