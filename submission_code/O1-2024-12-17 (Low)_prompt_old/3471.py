class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        averages = []
        # Repeat the process n//2 times
        for _ in range(len(nums)//2):
            min_element = min(nums)
            nums.remove(min_element)
            max_element = max(nums)
            nums.remove(max_element)
            averages.append((min_element + max_element) / 2)
        return min(averages)