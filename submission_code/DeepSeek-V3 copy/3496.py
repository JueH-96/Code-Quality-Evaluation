class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        left, right = 1, mountainHeight * (mountainHeight + 1) // 2 * max(workerTimes)
        
        def can_reduce(time):
            total = 0
            for t in workerTimes:
                # Find the maximum x such that t * (x * (x + 1)) // 2 <= time
                low, high = 0, mountainHeight
                best = 0
                while low <= high:
                    mid = (low + high) // 2
                    sum_x = t * mid * (mid + 1) // 2
                    if sum_x <= time:
                        best = mid
                        low = mid + 1
                    else:
                        high = mid - 1
                total += best
                if total >= mountainHeight:
                    return True
            return total >= mountainHeight
        
        while left < right:
            mid = (left + right) // 2
            if can_reduce(mid):
                right = mid
            else:
                left = mid + 1
        return left