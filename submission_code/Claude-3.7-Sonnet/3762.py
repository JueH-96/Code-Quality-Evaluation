class Solution:
    def maxScore(self, points: List[int], m: int) -> int:
        import math
        n = len(points)
        
        # Binary search to find the maximum threshold
        low, high = 0, min(points) * m
        
        while low <= high:
            mid = (low + high) // 2
            if self.can_achieve_threshold(points, m, mid):
                low = mid + 1
            else:
                high = mid - 1
        
        return high

    def can_achieve_threshold(self, points, m, threshold):
        n = len(points)
        # For each index, calculate how many visits needed to reach threshold
        required_visits = [math.ceil(threshold / p) for p in points]
        
        # In a zigzag pattern, interior indices are visited twice per complete zigzag
        # The rightmost index is visited only once per zigzag
        visits_per_zigzag = [2] * n
        visits_per_zigzag[n-1] = 1
        
        # Calculate how many zigzags needed for each index
        zigzags_required = [math.ceil(required_visits[i] / visits_per_zigzag[i]) for i in range(n)]
        
        # We need enough zigzags to satisfy all indices
        max_zigzags_required = max(zigzags_required)
        
        # Calculate total moves needed:
        # 1 move to first enter the array + moves for all required zigzags
        moves_required = 1 + 2 * (n - 1) * max_zigzags_required
        
        return moves_required <= m