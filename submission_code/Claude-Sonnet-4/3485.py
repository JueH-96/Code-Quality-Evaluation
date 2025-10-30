class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        # Sort intervals by start point
        start.sort()
        n = len(start)
        
        def canAchieveMinDiff(minDiff):
            # Try to greedily place integers with at least minDiff apart
            lastChosen = start[0]  # Choose the leftmost point of first interval
            
            for i in range(1, n):
                # For interval [start[i], start[i] + d]
                # We need to choose a point that's at least minDiff away from lastChosen
                nextMin = lastChosen + minDiff
                
                # Check if this nextMin falls within the current interval
                if nextMin > start[i] + d:
                    return False
                
                # Choose the maximum of nextMin and start[i] (leftmost point of current interval)
                lastChosen = max(nextMin, start[i])
            
            return True
        
        # Binary search on the answer
        left, right = 0, start[-1] + d - start[0]
        result = 0
        
        while left <= right:
            mid = (left + right) // 2
            
            if canAchieveMinDiff(mid):
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return result