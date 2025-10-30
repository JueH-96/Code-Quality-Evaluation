class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        n = len(nums)
        nums.sort()
        if n == 0 or nums[0] != 1:
            return 1
        for i in range(1, n):
            if nums[i] != nums[i-1] + 1:
                return nums[i] - 1
        return nums[-1] + 1