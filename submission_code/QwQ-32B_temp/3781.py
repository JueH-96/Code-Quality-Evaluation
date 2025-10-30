from typing import List

class Solution:
    def maxDistance(self, side: int, points: List[List[int]], k: int) -> int:
        def compute_t(x, y):
            if y == 0:
                return x
            elif x == side:
                return side + y
            elif y == side:
                return 2 * side + (side - x)
            else:  # x == 0 and y < side (left edge)
                return 3 * side + (side - y)
        
        ts = [compute_t(x, y) for x, y in points]
        ts.sort()
        
        low = 0
        high = 4 * side
        ans = 0
        
        while low <= high:
            mid = (low + high) // 2
            if self.is_possible(mid, ts, k, side):
                ans = mid
                low = mid + 1
            else:
                high = mid - 1
        return ans
    
    def is_possible(self, D, ts, k, side):
        if 4 * side < D * k:
            return False
        selected = 0
        prev = -float('inf')
        for t in ts:
            if t - prev >= D:
                selected += 1
                prev = t
                if selected == k:
                    return True
        return False