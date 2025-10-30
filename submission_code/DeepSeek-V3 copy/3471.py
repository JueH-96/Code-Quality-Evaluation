class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        averages = []
        while len(nums) > 0:
            minElement = min(nums)
            maxElement = max(nums)
            nums.remove(minElement)
            nums.remove(maxElement)
            average = (minElement + maxElement) / 2
            averages.append(average)
        return min(averages)