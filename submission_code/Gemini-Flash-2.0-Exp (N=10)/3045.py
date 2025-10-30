class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 1:
            return 0
        
        for shifts in range(n):
            shifted_nums = nums[-shifts:] + nums[:-shifts]
            is_sorted = True
            for i in range(n - 1):
                if shifted_nums[i] > shifted_nums[i+1]:
                    is_sorted = False
                    break
            if is_sorted:
                return shifts
        
        return -1