import typing
from typing import List

class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        sorted_start = sorted(start)
        
        def feasible(S: int) -> bool:
            prev = sorted_start[0]
            for num in sorted_start[1:]:
                next_val = max(num, prev + S)
                if next_val > num + d:
                    return False
                prev = next_val
            return True
        
        left, right = 0, sorted_start[-1] + d - sorted_start[0]
        while left <= right:
            mid = (left + right) // 2
            if feasible(mid):
                left = mid + 1
            else:
                right = mid - 1
        
        return right