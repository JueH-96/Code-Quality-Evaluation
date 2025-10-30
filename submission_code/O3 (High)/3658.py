from typing import List

class Solution:
    def minDifference(self, nums: List[int]) -> int:
        n = len(nums)

        # d0 – maximal adjacent difference of already known numbers
        d0 = 0
        for i in range(1, n):
            if nums[i] != -1 and nums[i-1] != -1:
                d0 = max(d0, abs(nums[i] - nums[i-1]))

        # --------------------------------------------------------------------
        # Collect information about every block of consecutive “-1”.
        # For every block we remember:
        #   • only one known neighbour   -> store that single value
        #   • two known neighbours
        #          • length 1            -> store the pair as a “short” block
        #          • length ≥ 2          -> store the two values as a “long” block
        # All of that information is needed each time we check a candidate
        # answer so we store it once and reuse it.
        # --------------------------------------------------------------------
        singles      = []          # blocks that touch just one known value
        short_pairs  = []          # blocks of length 1 between two known values
        long_pairs   = []          # blocks of length ≥2 between two known values

        i = 0
        while i < n:
            if nums[i] != -1:
                i += 1
                continue

            j = i
            while j < n and nums[j] == -1:
                j += 1          # [i, j) is the block of -1's

            left  = nums[i - 1] if i > 0 else None
            right = nums[j]     if j < n else None
            length = j - i

            if left is None and right is None:
                # whole array is -1
                return 0
            elif left is None:
                singles.append(right)
            elif right is None:
                singles.append(left)
            else:
                if length == 1:
                    short_pairs.append((left, right))
                else:
                    long_pairs.append((left, right))

            i = j

        # --------------------------------------------------------------------
        # Helper that checks if a given value M is achievable.
        # --------------------------------------------------------------------
        def feasible(M: int) -> bool:
            intervals = []          # intervals that must contain at least one centre
            bridging  = False       # True ⇔ at least one long block needs two centres

            # 1) blocks that touch a single known value
            for v in singles:
                intervals.append((v - M, v + M))

            # 2) blocks of length 1 – both neighbours must be the very same centre
            for a, b in short_pairs:
                if abs(a - b) > 2 * M:
                    return False
                if a > b:
                    a, b = b, a
                intervals.append((b - M, a + M))       # intersection of the two discs

            # 3) long blocks: every end may use its own centre
            for a, b in long_pairs:
                intervals.append((a - M, a + M))
                intervals.append((b - M, b + M))
                if abs(a - b) > 2 * M:                 # ends cannot share one centre
                    bridging = True                    # …so centres must be “close”

            # ------------------------------------------
            # cover all intervals with two points
            # ------------------------------------------
            if not intervals:
                return True            # only d0 matters – handled outside

            intervals.sort(key=lambda x: x[1])         # sort by right border

            # greedy – no distance restriction
            p1 = intervals[0][1]
            k  = 1
            while k < len(intervals) and intervals[k][0] <= p1 <= intervals[k][1]:
                k += 1
            if k == len(intervals):
                # one point already covers everything
                return True if not bridging else False

            p2 = intervals[k][1]
            k  += 1
            while k < len(intervals) and (intervals[k][0] <= p2 <= intervals[k][1]
                                          or intervals[k][0] <= p1 <= intervals[k][1]):
                k += 1

            if k < len(intervals):
                # even without distance bound we cannot cover – impossible
                return False

            if not bridging:
                return True            # distance between p1 and p2 free if no long pair needs both

            # ----------------------------------------------------------------
            # “bridging” == True  ➜  we additionally need |p1−p2| ≤ M
            # try to place two points within distance M that still hit all intervals
            # ----------------------------------------------------------------
            intervals.sort(key=lambda x: x[1])         # again by right border
            anchor = intervals[0][1]                   # take rightmost point of first
            lo = anchor - M
            hi = anchor + M
            for l, r in intervals[1:]:
                if l <= anchor <= r:
                    continue                           # already hit by anchor
                lo = max(lo, l)
                hi = min(hi, r)
                if lo > hi:
                    return False                       # cannot place second within all demands
            return True

        # --------------------------------------------------------------------
        # binary search for the minimal possible maximum difference
        # --------------------------------------------------------------------
        left, right = d0, 10 ** 9
        while left < right:
            mid = (left + right) // 2
            if feasible(mid):
                right = mid
            else:
                left = mid + 1
        return left