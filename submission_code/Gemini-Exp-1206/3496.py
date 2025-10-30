class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        def time_needed(worker_time, height):
            return worker_time * height * (height + 1) // 2

        def can_finish(time):
            total_height_reduced = 0
            for worker_time in workerTimes:
                low = 0
                high = mountainHeight
                height_reduced = 0
                while low <= high:
                    mid = (low + high) // 2
                    if time_needed(worker_time, mid) <= time:
                        height_reduced = mid
                        low = mid + 1
                    else:
                        high = mid - 1
                total_height_reduced += height_reduced
            return total_height_reduced >= mountainHeight

        low = 0
        high = max(workerTimes) * mountainHeight * (mountainHeight + 1) // 2
        ans = high
        while low <= high:
            mid = (low + high) // 2
            if can_finish(mid):
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        return ans