class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        averages = []
        n = len(nums)
        for _ in range(n // 2):
            current_min = min(nums)
            current_max = max(nums)
            nums.remove(current_min)
            nums.remove(current_max)
            averages.append((current_min + current_max) / 2)
        return min(averages)