class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        drops = 0
        di = -1
        for i in range(n - 1):
            if nums[i] > nums[i + 1]:
                drops += 1
                di = i
        if drops == 0:
            return 0
        if drops == 1 and nums[-1] < nums[0]:
            return n - (di + 1)
        return -1