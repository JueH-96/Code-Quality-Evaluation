class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        sorted_nums = sorted(nums)
        for i in range(n):
            shifted = nums[i:] + nums[:i]
            if shifted == sorted_nums:
                return i
        return -1