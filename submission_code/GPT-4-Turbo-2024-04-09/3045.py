class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        sorted_nums = sorted(nums)
        n = len(nums)
        
        for i in range(n):
            # Generate the array after i right shifts
            shifted = nums[-i:] + nums[:-i]
            if shifted == sorted_nums:
                return i
        
        return -1