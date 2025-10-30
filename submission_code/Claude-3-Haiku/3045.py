class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Check if the array is already sorted
        if nums == sorted(nums):
            return 0
        
        # Check if it's possible to sort the array using right shifts
        for i in range(n):
            if (i + 1) % n != nums[i]:
                return -1
        
        # Find the minimum number of right shifts required
        shifts = 0
        while True:
            sorted_nums = sorted(nums)
            if nums == sorted_nums:
                return shifts
            
            # Perform a right shift
            nums = [nums[-1]] + nums[:-1]
            shifts += 1