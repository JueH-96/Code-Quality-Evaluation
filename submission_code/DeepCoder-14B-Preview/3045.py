class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        if n == 1:
            return 0
        
        count = 0
        drop_index = -1
        
        for i in range(n):
            next_i = (i + 1) % n
            if nums[i] > nums[next_i]:
                count += 1
                drop_index = i
        
        if count != 1:
            return -1
        
        shifts = n - drop_index - 1
        
        # Simulate the shifts and check if the array is sorted
        shifted = nums[-shifts:] + nums[:-shifts] if shifts != 0 else nums.copy()
        
        # Check if the shifted array is sorted
        for i in range(len(shifted) - 1):
            if shifted[i] > shifted[i + 1]:
                return -1
        
        return shifts