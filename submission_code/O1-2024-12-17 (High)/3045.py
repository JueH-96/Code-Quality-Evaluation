class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        sorted_nums = sorted(nums)
        
        # If already sorted, no shifts needed
        if nums == sorted_nums:
            return 0
        
        # Try all possible right shifts up to n-1
        rotated = nums[:]
        for shift_count in range(1, n):
            # Perform one right shift
            rotated = [rotated[-1]] + rotated[:-1]
            if rotated == sorted_nums:
                return shift_count
        
        # If none of the rotations result in a sorted array, return -1
        return -1