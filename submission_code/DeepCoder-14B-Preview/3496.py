class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        if not workerTimes:
            return 0
        
        def x_max(t, T):
            left = 1
            right = 2 * 10**5  # A sufficiently large upper bound
            best = 0
            while left <= right:
                mid = (left + right) // 2
                product = mid * (mid + 1) * t
                if product <= 2 * T:
                    best = mid
                    left = mid + 1
                else:
                    right = mid - 1
            return best
        
        max_t = max(workerTimes)
        high = max_t * mountainHeight * (mountainHeight + 1) // 2
        low = 0
        
        while low < high:
            mid = (low + high) // 2
            total = 0
            for t in workerTimes:
                total += x_max(t, mid)
                if total >= mountainHeight:
                    break  # No need to process further workers
            if total >= mountainHeight:
                high = mid
            else:
                low = mid + 1
        
        return low