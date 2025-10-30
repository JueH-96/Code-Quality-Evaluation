class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        n = len(nums)
        nums.sort()
        averages = []
        for i in range(n // 2):
            minElement = nums[0]
            maxElement = nums[-1]
            nums.pop(0)
            nums.pop()
            averages.append((minElement + maxElement) / 2)
        return min(averages)