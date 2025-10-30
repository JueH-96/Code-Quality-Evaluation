class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        averages = []
        while nums:
            minElement = min(nums)
            maxElement = max(nums)
            nums.remove(minElement)
            nums.remove(maxElement)
            averages.append((minElement + maxElement) / 2)
        return min(averages)