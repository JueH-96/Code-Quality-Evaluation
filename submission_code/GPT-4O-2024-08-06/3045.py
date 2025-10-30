class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Check if the array is already sorted
        if nums == sorted(nums):
            return 0
        
        # Find the index where the array can be split into two sorted parts
        for i in range(n):
            # Check if the array from i to end and from start to i is sorted
            if nums[i:] + nums[:i] == sorted(nums):
                return n - i
        
        # If no such index is found, return -1
        return -1