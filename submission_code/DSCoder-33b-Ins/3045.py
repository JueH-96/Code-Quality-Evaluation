class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        sorted_nums = sorted(nums)
        n = len(nums)
        for i in range(n):
            if nums[i:] + nums[:i] == sorted_nums:
                return i
        return -1