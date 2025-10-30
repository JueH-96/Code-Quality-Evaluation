class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        import math
        
        def canFinish(time):
            total_height = 0
            for workerTime in workerTimes:
                k = 2 * time / workerTime
                max_x = int((-1 + math.sqrt(1 + 4 * k)) / 2)
                total_height += max_x
                if total_height >= mountainHeight:
                    return True
            return False
        
        left, right = 1, max(workerTimes) * mountainHeight * (mountainHeight + 1) // 2
        
        while left < right:
            mid = (left + right) // 2
            if canFinish(mid):
                right = mid
            else:
                left = mid + 1
        
        return left