from typing import List

class Solution:
    def maxScore(self, points: List[int], m: int) -> int:
        n = len(points)
        if n == 0:
            return 0

        def is_possible(X: int) -> bool:
            k = [(X + p - 1) // p for p in points]
            visits = [0] * n
            visits[0] = 1  # initial move to index 0
            moves_used = 1
            if moves_used > m:
                return False
            i = 0
            while i < n:
                needed = k[i] - visits[i]
                if needed <= 0:
                    i += 1
                    continue
                if i == n - 1:
                    return False
                additional_moves = 2 * needed
                if moves_used + additional_moves > m:
                    return False
                moves_used += additional_moves
                visits[i] += needed
                visits[i + 1] += needed
                i += 1
            return visits[-1] >= k[-1]

        low = 0
        high = max(points) * m
        ans = 0
        while low <= high:
            mid = (low + high) // 2
            if is_possible(mid):
                ans = mid
                low = mid + 1
            else:
                high = mid - 1
        return ans