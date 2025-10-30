class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        nums.sort()
        averages = []
        for _ in range(len(nums) // 2):
            minElement = nums.pop(0)
            maxElement = nums.pop(-1)
            averages.append((minElement + maxElement) / 2)
        return min(averages)