class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        averages = []
        nums.sort()
        for _ in range(len(nums) // 2):
            minElement = nums.pop(0)
            maxElement = nums.pop()
            averages.append((minElement + maxElement) / 2)
        return min(averages)