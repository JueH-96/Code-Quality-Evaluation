class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        nums.sort()
        averages = []
        while len(nums) > 0:
            min_element = nums.pop(0)
            max_element = nums.pop()
            average = (min_element + max_element) / 2
            averages.append(average)
        return min(averages)