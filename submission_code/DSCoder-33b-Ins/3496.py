class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        def calculate_time(worker_time: int, mountain_height: int) -> int:
            quotient, remainder = divmod(mountain_height, worker_time)
            return quotient * worker_time + (quotient + (remainder > 0)) * worker_time

        return max(calculate_time(worker_time, mountainHeight) for worker_time in workerTimes)