class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        sorted_nums = sorted(nums)

        for i in range(n):
            if nums == sorted_nums:
                return i
            nums = [nums[-1]] + nums[:-1]

        return -1