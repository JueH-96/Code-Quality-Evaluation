from typing import List

class Solution:
    def maxScore(self, points: List[int], m: int) -> int:
        n = len(points)
        # check if we can achieve minimum score T in all gameScore[i]
        def can(T: int) -> bool:
            # if T == 0, always possible without any moves
            if T == 0:
                return True
            sum_r = 0      # total required visits
            cnt = 0        # count of i with at least one required visit
            max_i = -1     # rightmost index needing a visit
            # for each position compute r_i = ceil(T / points[i])
            for i, p in enumerate(points):
                # how many times we must visit i
                # we need gameScore[i] >= T, and each visit adds p
                r = (T + p - 1) // p
                if r > 0:
                    sum_r += r
                    cnt += 1
                    max_i = i
                    # if total required visits already exceed total moves m, fail
                    if sum_r > m:
                        return False
            # if no visits are required, T=0 was handled, but here T>0 => cnt>0
            # extra visits beyond the first at each index
            extras = sum_r - cnt
            # compute minimal moves:
            #  - initial sweep from -1 to max_i costs (max_i + 1) moves
            #  - each extra visit costs 2 moves (you must step out and back)
            #  - we can schedule the very last extra so that we don't pay one return move
            #    (i.e. save 1 move if there's at least one extra)
            moves_needed = (max_i + 1) + 2 * extras
            if extras > 0:
                moves_needed -= 1
            return moves_needed <= m

        # binary search on T
        lo, hi = 0, max(points) * m
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if can(mid):
                lo = mid
            else:
                hi = mid - 1
        return lo

# Example usage:
# sol = Solution()
# print(sol.maxScore([2,4], 3))  # expected 4
# print(sol.maxScore([1,2,3], 5))  # expected 2