import bisect
from typing import List

class Solution:
    def maxDistance(self, side: int, points: List[List[int]], k: int) -> int:
        def get_s(point):
            x, y = point
            if y == 0:
                return x
            elif x == side:
                return side + y
            elif y == side:
                return 3 * side - x
            else:  # x == 0, left edge (y != 0 and y != side)
                return 4 * side - y
        
        s = [get_s(p) for p in points]
        s.sort()
        n = len(s)
        L = 4 * side
        low = 0
        high = L
        best = 0
        
        def check(D):
            # Precompute the extended array S
            S = s + [x + L for x in s]
            for i in range(n):
                count = 1
                last_val = s[i]
                target = last_val + D
                current_pos = i
                # Attempt to select k points starting at i
                for _ in range(k-1):
                    # Binary search in S[i+1 ... i + n] for first element >= target
                    idx = bisect.bisect_left(S, target, i + 1, i + n + 1)
                    if idx > i + n:
                        break  # No valid element in the window
                    # Update state
                    count += 1
                    current_pos = idx
                    last_val = S[current_pos]
                    target = last_val + D
                # Check wrap-around condition
                if count >= k and (last_val + D <= s[i] + L):
                    return True
            return False
        
        while low <= high:
            mid = (low + high) // 2
            if check(mid):
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        return best