import math

class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        H = mountainHeight
        workers = workerTimes
        if not workers:
            return 0
        
        max_w = max(workers)
        high = (max_w * H * (H + 1)) // 2
        low = 0
        answer = high
        
        while low <= high:
            mid = (low + high) // 2
            total = 0
            for w in workers:
                if mid == 0:
                    x_i = 0
                else:
                    K = 2 * mid / w
                    if K < 2:
                        x_i = 0
                    else:
                        discriminant = 1 + 4 * K
                        sqrt_discriminant = math.sqrt(discriminant)
                        x = (sqrt_discriminant - 1) / 2
                        x_i = int(x)
                        # Adjust x_i if necessary
                        while x_i >= 0 and x_i * (x_i + 1) > K:
                            x_i -= 1
                total += x_i
                if total >= H:
                    break  # Early exit if sum exceeds H
            if total >= H:
                answer = mid
                high = mid - 1
            else:
                low = mid + 1
        return answer