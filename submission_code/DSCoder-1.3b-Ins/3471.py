class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        n = len(nums)
        averages = []
        for _ in range(n // 2):
            min_num = min(nums)
            max_num = max(nums)
            nums.remove(min_num)
            nums.remove(max_num)
            averages.append((min_num + max_num) / 2)
        return min(averages)