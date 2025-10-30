class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: list[int]) -> int:
        workerTimes.sort()
        while mountainHeight > len(workerTimes):
            for i in range(len(workerTimes)):
                workerTimes[i] += workerTimes[i] * (mountainHeight // len(workerTimes))
            mountainHeight %= len(workerTimes)
        for i in range(mountainHeight):
            workerTimes[i] += workerTimes[i] * (mountainHeight // len(workerTimes))
        return max(workerTimes)