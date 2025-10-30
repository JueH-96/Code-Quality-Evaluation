import math
from typing import List

class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        H = mountainHeight
        if H == 0:
            return 0
        
        # Compute the initial upper bound for binary search
        max_time = 0
        for w in workerTimes:
            current = w * H * (H + 1) // 2
            if current > max_time:
                max_time = current
        
        low = 0
        high = max_time
        ans = high
        
        while low <= high:
            mid = (low + high) // 2
            total = 0
            for w in workerTimes:
                if w == 0:
                    continue  # According to constraints, w >= 1
                
                K = 8 * mid / w
                sqrt_val = math.sqrt(1 + K)
                x = (sqrt_val - 1) / 2
                x_floor = int(x)
                
                # Check if x_floor + 1 is possible to handle precision issues
                next_x = x_floor + 1
                if next_x * (next_x + 1) * w <= 2 * mid:
                    x_floor = next_x
                
                total += x_floor
                if total >= H:
                    break  # Early exit if possible
            
            if total >= H:
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        
        return ans