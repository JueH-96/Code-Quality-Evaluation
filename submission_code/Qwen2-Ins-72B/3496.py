class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        n = len(workerTimes)
        max_time = 0
        for i in range(n):
            time = 0
            height = mountainHeight
            while height > 0:
                time += workerTimes[i]
                height -= 1
                if time > max_time:
                    max_time = time
                    break
            if height <= 0:
                max_time = max(max_time, time)
        return max_time