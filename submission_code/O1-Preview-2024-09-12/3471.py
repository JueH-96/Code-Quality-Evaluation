class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        from collections import deque
        nums.sort()
        nums = deque(nums)
        averages = []
        while nums:
            minElement = nums.popleft()
            maxElement = nums.pop()
            averages.append((minElement + maxElement) / 2)
        return min(averages)