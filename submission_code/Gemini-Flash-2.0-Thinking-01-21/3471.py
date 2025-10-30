class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        averages = []
        current_nums = list(nums) # Create a copy to avoid modifying the original list
        n = len(current_nums)
        for _ in range(n // 2):
            min_element = min(current_nums)
            max_element = max(current_nums)
            averages.append((min_element + max_element) / 2)
            current_nums.remove(min_element)
            current_nums.remove(max_element)
        return min(averages)