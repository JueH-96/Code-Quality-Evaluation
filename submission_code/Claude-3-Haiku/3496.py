class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        total_time = 0
        for worker_time in workerTimes:
            time_taken = 0
            for i in range(1, mountainHeight + 1):
                time_taken += worker_time * i
            total_time = max(total_time, time_taken)
        return total_time