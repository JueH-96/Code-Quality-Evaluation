class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        res = 1
        for i in range(1, n):
            if nums[i] - nums[i-1] > 1:
                nums[i] = nums[i-1] + 1
                res += 1
        return res