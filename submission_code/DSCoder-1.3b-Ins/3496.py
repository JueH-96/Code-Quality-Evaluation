class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        max_height = max(mountainHeight - i for i in workerTimes)
        return max_height