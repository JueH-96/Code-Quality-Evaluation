class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        workerTimes.sort()
        total_time = 0
        for i in range(len(workerTimes)):
            total_time += workerTimes[i] * (mountainHeight - i)
        return total_time