import math

class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        def is_possible(max_time: int) -> bool:
            total_height = 0
            for t in workerTimes:
                if t == 0:
                    continue  # avoid division by zero
                temp = 1 + 8 * max_time / t
                sqrt_temp = math.sqrt(temp)
                h_i = math.floor((-1 + sqrt_temp) / 2)
                total_height += h_i
                if total_height >= mountainHeight:
                    return True
            return total_height >= mountainHeight

        if mountainHeight == 0:
            return 0

        max_t = max(workerTimes)
        # Calculate high as the time for one worker to reduce the entire height
        high = max_t * mountainHeight * (mountainHeight + 1) // 2
        low = 0

        while low < high:
            mid = (low + high) // 2
            if is_possible(mid):
                high = mid
            else:
                low = mid + 1

        return low