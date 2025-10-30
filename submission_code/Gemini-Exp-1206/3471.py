class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        nums.sort()
        averages = []
        for i in range(len(nums) // 2):
            averages.append((nums[i] + nums[len(nums) - 1 - i]) / 2)
        return min(averages)