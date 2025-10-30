from typing import List

class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        start.sort()
        n = len(start)
        low = 0
        high = (start[-1] + d) - start[0]
        ans = 0
        
        while low <= high:
            mid = (low + high) // 2
            possible = True
            prev = -float('inf')
            for s in start:
                current_end = s + d
                candidate = max(s, prev + mid)
                if candidate > current_end:
                    possible = False
                    break
                prev = candidate
            if possible:
                ans = mid
                low = mid + 1
            else:
                high = mid - 1
        
        return ans