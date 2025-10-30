class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        for k in range(n):
            shifted = nums[-k:] + nums[:-k] if k != 0 else nums.copy()
            if all(shifted[i] <= shifted[i+1] for i in range(n-1)):
                return k
        return -1