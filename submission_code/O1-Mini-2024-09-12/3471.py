class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        nums_sorted = sorted(nums)
        averages = []
        while nums_sorted:
            min_element = nums_sorted.pop(0)
            max_element = nums_sorted.pop(-1)
            avg = (min_element + max_element) / 2
            averages.append(avg)
        return min(averages)