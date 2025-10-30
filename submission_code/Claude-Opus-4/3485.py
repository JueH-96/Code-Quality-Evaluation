class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        # Sort intervals by start point
        start.sort()
        n = len(start)
        
        def canAchieve(minDist):
            # Try to place integers with at least minDist between consecutive ones
            # Place first integer at the start of first interval
            prev = start[0]
            
            for i in range(1, n):
                # Next integer must be at least prev + minDist
                next_pos = max(start[i], prev + minDist)
                
                # Check if this position is within the interval
                if next_pos > start[i] + d:
                    return False
                
                prev = next_pos
            
            return True
        
        # Binary search on the answer
        left, right = 0, start[-1] + d - start[0]
        
        while left < right:
            mid = (left + right + 1) // 2
            if canAchieve(mid):
                left = mid
            else:
                right = mid - 1
        
        return left