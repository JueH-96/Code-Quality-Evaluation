class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        sorted_nums = sorted(nums)
        n = len(nums)
        return min((sorted_nums[i] + sorted_nums[n-1-i]) / 2.0 for i in range(n // 2))