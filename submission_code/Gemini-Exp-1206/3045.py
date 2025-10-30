class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        for i in range(n):
            if all(nums[j] < nums[(j + 1) % n] for j in range(n - 1)):
                return i
            nums = [nums[-1]] + nums[:-1]
        return -1