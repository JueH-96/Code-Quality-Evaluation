class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        workerTimes.sort()
        n = len(workerTimes)
        total_time = 0
        for i in range(n):
            x = mountainHeight // (n - i)
            total_time = max(total_time, workerTimes[i] * (x + (x * (x - 1)) // 2))
        return total_time