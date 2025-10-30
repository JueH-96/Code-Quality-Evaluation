class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        sorted_nums = sorted(nums)
        n = len(nums)
        
        # Try all possible k from 0..n-1
        for k in range(n):
            # A right shift by k is nums[-k:] + nums[:-k]
            candidate = nums[-k:] + nums[:-k]
            if candidate == sorted_nums:
                return k
        
        return -1