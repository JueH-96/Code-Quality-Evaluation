class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        averages = []
        nums_sorted = sorted(nums)
        while len(nums_sorted) > 0:
            min_element = nums_sorted.pop(0)
            max_element = nums_sorted.pop(-1)
            average = (min_element + max_element) / 2
            averages.append(average)
        return min(averages)