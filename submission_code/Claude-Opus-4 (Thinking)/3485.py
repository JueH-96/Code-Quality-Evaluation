class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        n = len(start)
        
        # Sort intervals by start point
        start.sort()
        
        def canAchieve(minDiff):
            # Try to place integers with minimum difference of minDiff
            prev = start[0]  # Choose the leftmost point of the first interval
            
            for i in range(1, n):
                # We need to place an integer at least minDiff away from prev
                # and within [start[i], start[i] + d]
                next_val = max(prev + minDiff, start[i])
                
                if next_val > start[i] + d:
                    # Can't place an integer in this interval
                    return False
                
                prev = next_val
            
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