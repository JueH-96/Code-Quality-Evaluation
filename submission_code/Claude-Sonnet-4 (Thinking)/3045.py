class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        sorted_nums = sorted(nums)
        
        for shifts in range(n):
            rotated = nums[-shifts:] + nums[:-shifts]
            if rotated == sorted_nums:
                return shifts
        
        return -1