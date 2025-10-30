class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Check if already sorted
        if nums == sorted(nums):
            return 0
        
        # Find the break point where nums[i] > nums[i+1]
        break_point = -1
        for i in range(n - 1):
            if nums[i] > nums[i + 1]:
                if break_point == -1:
                    break_point = i
                else:
                    # More than one break point means impossible to sort
                    return -1
        
        # If no break point found but array isn't sorted, impossible
        if break_point == -1:
            return -1
        
        # Check if rotation at break_point + 1 gives sorted array
        # The rotated array would be nums[break_point+1:] + nums[:break_point+1]
        rotated = nums[break_point + 1:] + nums[:break_point + 1]
        
        if rotated == sorted(nums):
            # Number of right shifts needed = n - (break_point + 1)
            return n - (break_point + 1)
        else:
            return -1