class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        sorted_nums = sorted((val, i) for i, val in enumerate(nums))
        min_diff = float('inf')
        for i in range(x, len(nums)):
            min_diff = min(min_diff, abs(sorted_nums[i][0] - sorted_nums[i - x][0]))
        return min_diff