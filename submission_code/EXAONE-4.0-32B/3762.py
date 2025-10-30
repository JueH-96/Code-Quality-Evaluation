class Solution:
    def maxScore(self, points: List[int], m: int) -> int:
        n = len(points)
        if m < n:
            return 0
        extra = m - n
        low, high = 1, max(points) * m
        
        while low <= high:
            mid = (low + high) // 2
            need = 0
            for i in range(n):
                ceil_val = (mid + points[i] - 1) // points[i]
                need += ceil_val - 1
                if need > extra:
                    break
            
            if need <= extra:
                low = mid + 1
            else:
                high = mid - 1
        
        return high