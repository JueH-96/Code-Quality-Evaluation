class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        def getTimeForHeight(workerTime: int, height: int) -> int:
            total = 0
            for i in range(1, height + 1):
                total += workerTime * i
            return total
            
        def canFinishInTime(seconds: int) -> bool:
            totalHeight = 0
            for workerTime in workerTimes:
                # For each worker, find maximum height they can reduce in given seconds
                left, right = 0, mountainHeight
                maxHeight = 0
                
                while left <= right:
                    mid = (left + right) // 2
                    timeNeeded = getTimeForHeight(workerTime, mid)
                    
                    if timeNeeded <= seconds:
                        maxHeight = mid
                        left = mid + 1
                    else:
                        right = mid - 1
                        
                totalHeight += maxHeight
                
                if totalHeight >= mountainHeight:
                    return True
                    
            return False
            
        # Binary search on answer
        left, right = 0, mountainHeight * max(workerTimes) * (mountainHeight + 1) // 2
        result = right
        
        while left <= right:
            mid = (left + right) // 2
            
            if canFinishInTime(mid):
                result = mid
                right = mid - 1
            else:
                left = mid + 1
                
        return result