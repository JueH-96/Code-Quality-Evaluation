import bisect
from typing import List

class Solution:
    def maxDistance(self, side: int, points: List[List[int]], k: int) -> int:
        L = 4 * side
        n = len(points)
        positions = []
        for p in points:
            x, y = p
            if x == 0:
                pos = y
            elif y == side:
                pos = side + x
            elif x == side:
                pos = 3 * side - y
            else:
                pos = 4 * side - x
            positions.append(pos)
        
        positions.sort()
        positions2 = positions + [p + L for p in positions]
        
        low, high = 0, 2 * side
        ans = 0
        
        while low <= high:
            mid = (low + high) // 2
            if self.checker(mid, k, L, n, positions2):
                ans = mid
                low = mid + 1
            else:
                high = mid - 1
                
        return ans

    def checker(self, d: int, k: int, L: int, n: int, positions2: List[int]) -> bool:
        if k == 1:
            return True
        if k * d > L:
            return False
            
        for start in range(n):
            count = 1
            current = positions2[start]
            next_search_start = start + 1
            found = True
            for _ in range(k - 1):
                j = bisect.bisect_left(positions2, current + d, lo=next_search_start, hi=start + n)
                if j == start + n:
                    found = False
                    break
                count += 1
                current = positions2[j]
                next_search_start = j + 1
                
            if not found or count < k:
                continue
                
            linear_dist = current - positions2[start]
            circular_dist = min(linear_dist, L - linear_dist)
            if circular_dist >= d:
                return True
                
        return False