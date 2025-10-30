class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        nums_sorted = sorted(nums)
        averages = []
        n = len(nums_sorted)
        for i in range(n // 2):
            min_elem = nums_sorted[i]
            max_elem = nums_sorted[n - 1 - i]
            averages.append((min_elem + max_elem) / 2)
        return min(averages)