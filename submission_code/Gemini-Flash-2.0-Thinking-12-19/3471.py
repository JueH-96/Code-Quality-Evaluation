class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        averages = []
        nums_list = list(nums) # Create a mutable copy of nums
        n = len(nums_list)
        for _ in range(n // 2):
            nums_list.sort()
            min_element = nums_list.pop(0)
            max_element = nums_list.pop(-1)
            averages.append((min_element + max_element) / 2)
        return min(averages)