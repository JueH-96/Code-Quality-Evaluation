class Solution:
    def maxScore(self, points: List[int], m: int) -> int:
        n = len(points)
        
        def canAchieve(minScore):
            moves = 0
            left = 0
            right = n - 1
            
            while left <= right:
                if points[left] < minScore:
                    moves += (minScore - points[left])
                if points[right] < minScore:
                    moves += (minScore - points[right])
                left += 1
                right -= 1
            
            return moves <= m
        
        low, high = 0, max(points)
        result = 0
        
        while low <= high:
            mid = (low + high) // 2
            if canAchieve(mid):
                result = mid
                low = mid + 1
            else:
                high = mid - 1
        
        return result