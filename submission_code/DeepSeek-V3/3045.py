class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        sorted_nums = sorted(nums)
        if nums == sorted_nums:
            return 0
        for shift in range(1, n):
            shifted = nums[-shift:] + nums[:-shift]
            if shifted == sorted_nums:
                return shift
        return -1