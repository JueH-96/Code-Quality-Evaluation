class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        shifts = [0]*n
        for i in range(n):
            shifts[i] = nums[i]
        shifts.sort()
        if shifts[0] != nums[0]:
            return -1
        else:
            shifts = shifts[1:]
            shifts.append(shifts[0])
            return sum(shifts)