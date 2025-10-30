class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        if mountainHeight == 0:
            return 0
        
        workerTimes.sort()
        max_worker = max(workerTimes) if workerTimes else 0
        high = max_worker * mountainHeight * (mountainHeight + 1) // 2
        low = 0
        
        while low < high:
            mid = (low + high) // 2
            total = 0
            for w in workerTimes:
                if total >= mountainHeight:
                    break
                left, right = 0, 2 * 10**9  # Arbitrarily large upper bound
                max_x = 0
                while left <= right:
                    m = (left + right) // 2
                    time_needed = w * m * (m + 1) // 2
                    if time_needed <= mid:
                        max_x = m
                        left = m + 1
                    else:
                        right = m - 1
                total += max_x
            if total >= mountainHeight:
                high = mid
            else:
                low = mid + 1
        
        return low