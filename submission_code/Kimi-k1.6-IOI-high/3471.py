class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        nums.sort()
        n = len(nums)
        averages = [(nums[i] + nums[n - 1 - i]) / 2 for i in range(n // 2)]
        return min(averages)