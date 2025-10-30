class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        # Check if the array is already sorted
        if nums == sorted(nums):
            return 0
        
        # Find the index where the sorted order breaks
        for i in range(n-1):
            if nums[i] > nums[i+1]:
                break
        else:
            # If the loop completes without finding a break, the array is in reverse sorted order
            return -1
        
        # Check if the array can be sorted by right shifts
        if nums[i:] + nums[:i] == sorted(nums):
            return n - i
        else:
            return -1