import math

class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        min_w = min(workerTimes)
        high = min_w * mountainHeight * (mountainHeight + 1) // 2
        low = 0
        
        def is_feasible(T):
            sum_x = 0
            for w in workerTimes:
                if w == 0:
                    continue
                temp = 8 * T / w
                sqrt_temp = math.sqrt(temp)
                x = (sqrt_temp - 1) / 2
                x_floor = int(x)
                # Check x_floor + 1
                if (x_floor + 1) * (x_floor + 2) * w <= 2 * T:
                    x_i = x_floor + 1
                else:
                    x_i = x_floor
                sum_x += x_i
                if sum_x >= mountainHeight:
                    break
            return sum_x >= mountainHeight
        
        while low < high:
            mid = (low + high) // 2
            if is_feasible(mid):
                high = mid
            else:
                low = mid + 1
        return low