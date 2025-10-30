class Solution:
    def countWays(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        count = 0
        if nums[0] > 0:
            count += 1
        for i in range(1, n):
            if nums[i] > i and nums[i] != nums[i-1]:
                count += 1
        if nums[-1] < n:
            count += 1
        return count