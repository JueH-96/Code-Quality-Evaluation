import math

class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        def check(time):
            total_reduction = 0
            for worker_time in workerTimes:
                if worker_time == 0:
                    continue
                h_i = int((-1 + math.sqrt(1 + 8 * time / worker_time)) / 2)
                total_reduction += h_i
            return total_reduction >= mountainHeight

        low = 0
        high = 10**12
        ans = high
        while low <= high:
            mid = (low + high) // 2
            if check(mid):
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        return ans