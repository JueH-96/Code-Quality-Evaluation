class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Check if already sorted
        if all(nums[i] < nums[i+1] for i in range(n-1)):
            return 0
        
        # Find the number of places where nums[i] > nums[i+1]
        break_index = -1
        break_count = 0
        for i in range(n-1):
            if nums[i] > nums[i+1]:
                break_count += 1
                break_index = i
        # Also check transition from last to first
        if nums[n-1] > nums[0]:
            break_count += 1
            break_index = n-1
        
        # If more than 1 break, cannot be sorted by rotation
        if break_count > 1:
            return -1
        
        # Calculate shifts needed
        shifts = n - (break_index + 1)
        
        # Perform the shifts (or check if it's sorted after that many shifts)
        # Rotate right 'shifts' times
        rotated = nums[-shifts:] + nums[:-shifts]
        
        # Check if rotation results in a sorted array
        if all(rotated[i] < rotated[i+1] for i in range(n-1)):
            return shifts
        else:
            return -1