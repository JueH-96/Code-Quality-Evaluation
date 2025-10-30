class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        sorted_nums = sorted(nums)
        
        # Check if the array can be sorted by right shifts
        if nums == sorted_nums:
            return 0
        
        # Find the index where the sorted array starts
        for shift in range(n):
            if all(nums[(i + shift) % n] == sorted_nums[i] for i in range(n)):
                return shift
        
        return -1