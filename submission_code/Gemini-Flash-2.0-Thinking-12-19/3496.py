class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        def get_reduction_height(time, worker_time):
            low = 0
            high = mountainHeight
            ans = 0
            while low <= high:
                mid = (low + high) // 2
                current_time = worker_time * mid * (mid + 1) // 2
                if current_time <= time:
                    ans = mid
                    low = mid + 1
                else:
                    high = mid - 1
            return ans

        def is_possible(time):
            total_reduction = 0
            for worker_time in workerTimes:
                total_reduction += get_reduction_height(time, worker_time)
            return total_reduction >= mountainHeight

        low = 0
        high = mountainHeight * max(workerTimes) * mountainHeight
        ans = high
        while low <= high:
            mid = (low + high) // 2
            if is_possible(mid):
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        return ans