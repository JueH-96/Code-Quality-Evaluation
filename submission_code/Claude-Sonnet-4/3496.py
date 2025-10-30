class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        def maxHeightInTime(workerTime, timeLimit):
            # Find maximum height this worker can reduce in timeLimit
            # Need to solve: workerTime * x * (x + 1) / 2 <= timeLimit
            # So: x * (x + 1) <= 2 * timeLimit / workerTime
            
            if timeLimit < workerTime:
                return 0
            
            # Binary search for maximum x
            left, right = 1, mountainHeight
            result = 0
            
            while left <= right:
                mid = (left + right) // 2
                timeNeeded = workerTime * mid * (mid + 1) // 2
                
                if timeNeeded <= timeLimit:
                    result = mid
                    left = mid + 1
                else:
                    right = mid - 1
            
            return result
        
        def canFinishInTime(timeLimit):
            totalHeight = 0
            for workerTime in workerTimes:
                totalHeight += maxHeightInTime(workerTime, timeLimit)
                if totalHeight >= mountainHeight:
                    return True
            return totalHeight >= mountainHeight
        
        # Binary search on the answer
        left = 1
        right = min(workerTimes) * mountainHeight * (mountainHeight + 1) // 2
        
        while left < right:
            mid = (left + right) // 2
            if canFinishInTime(mid):
                right = mid
            else:
                left = mid + 1
        
        return left