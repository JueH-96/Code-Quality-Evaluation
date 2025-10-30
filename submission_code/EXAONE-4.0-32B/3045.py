class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        for k in range(n):
            valid = True
            for i in range(n - 1):
                idx1 = (n - k + i) % n
                idx2 = (n - k + i + 1) % n
                if nums[idx1] > nums[idx2]:
                    valid = False
                    break
            if valid:
                return k
        return -1