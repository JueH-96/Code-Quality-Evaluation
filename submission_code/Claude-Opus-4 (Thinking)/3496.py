class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        def canFinish(time):
            total_reduced = 0
            for wt in workerTimes:
                # Binary search for max x such that wt * x * (x + 1) // 2 <= time
                left, right = 0, mountainHeight
                while left < right:
                    mid = (left + right + 1) // 2
                    if wt * mid * (mid + 1) // 2 <= time:
                        left = mid
                    else:
                        right = mid - 1
                total_reduced += left
                if total_reduced >= mountainHeight:
                    return True
            return False
        
        # Binary search on the answer
        left = min(workerTimes)
        right = max(workerTimes) * mountainHeight * (mountainHeight + 1) // 2
        
        while left < right:
            mid = (left + right) // 2
            if canFinish(mid):
                right = mid
            else:
                left = mid + 1
        
        return left