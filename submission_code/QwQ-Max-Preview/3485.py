from typing import List

class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        start.sort()
        n = len(start)
        
        def is_possible(g):
            prev = start[0]
            for i in range(1, n):
                current_start = start[i]
                current_end = current_start + d
                required_min = prev + g
                x_i = max(required_min, current_start)
                if x_i > current_end:
                    return False
                prev = x_i
            return True
        
        low = 0
        high = (start[-1] + d) - start[0]
        max_g = 0
        
        while low <= high:
            mid = (low + high) // 2
            if is_possible(mid):
                max_g = mid
                low = mid + 1
            else:
                high = mid - 1
        
        return max_g