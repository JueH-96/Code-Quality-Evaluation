class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        def canFinishInTime(maxTime):
            totalHeight = 0
            for workerTime in workerTimes:
                # Find maximum height this worker can reduce in maxTime
                # workerTime * x * (x + 1) / 2 <= maxTime
                # x * (x + 1) <= 2 * maxTime / workerTime
                # Using quadratic formula to find max x
                discriminant = 1 + 8 * maxTime / workerTime
                x = int((-1 + discriminant ** 0.5) / 2)
                totalHeight += x
                if totalHeight >= mountainHeight:
                    return True
            return totalHeight >= mountainHeight
        
        # Binary search on the answer
        left, right = 1, mountainHeight * (mountainHeight + 1) // 2 * max(workerTimes)
        
        while left < right:
            mid = (left + right) // 2
            if canFinishInTime(mid):
                right = mid
            else:
                left = mid + 1
        
        return left