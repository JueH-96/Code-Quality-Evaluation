class Solution:
    def maxScore(self, points: List[int], m: int) -> int:
        n = len(points)
        
        # Binary search on the minimum score
        left, right = 0, m * max(points)
        
        while left < right:
            mid = (left + right + 1) // 2
            
            # Check if we can achieve a minimum score of 'mid'
            total_visits = 0
            for p in points:
                visits_needed = (mid + p - 1) // p  # Ceiling division
                total_visits += visits_needed
            
            # If total visits <= m, we can achieve this minimum score
            if total_visits <= m:
                left = mid
            else:
                right = mid - 1
        
        return left