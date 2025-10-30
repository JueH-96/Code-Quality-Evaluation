class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        target = sorted(nums)
        n = len(nums)
        for k in range(n):
            rotated = nums[-k:] + nums[:-k]
            if rotated == target:
                return k
        return -1