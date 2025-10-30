from typing import List

class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        start.sort()
        n = len(start)
        if n == 0:
            return 0
        
        low = 0
        high = (start[-1] + d) - start[0]
        ans = 0
        
        while low <= high:
            mid = (low + high) // 2
            if self.is_possible(start, d, mid):
                ans = mid
                low = mid + 1
            else:
                high = mid - 1
        
        return ans
    
    def is_possible(self, S: List[int], d: int, D: int) -> bool:
        prev = S[0]
        for i in range(1, len(S)):
            current_start = S[i]
            earliest_x = max(current_start, prev + D)
            if earliest_x > current_start + d:
                return False
            prev = earliest_x
        return True