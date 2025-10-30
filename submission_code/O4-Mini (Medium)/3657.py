from typing import List
import bisect

class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        # Helper to check one-dimensional intervals for two cuts
        def check_1d(intervals):
            m = len(intervals)
            # Separate starts and ends
            starts = [l for l, r in intervals]
            ends = [r for l, r in intervals]
            starts.sort()
            ends.sort()
            # Build candidate cut positions Y = all unique l or r in (0, n)
            Y_set = set()
            for l, r in intervals:
                if 1 <= l <= n-1:
                    Y_set.add(l)
                if 1 <= r <= n-1:
                    Y_set.add(r)
            if not Y_set:
                return False
            Y = sorted(Y_set)
            # Build sweep events for intervals that strictly cover some integer y:
            # we mark +1 at (l+1) and -1 at r for any interval with l+1 <= r-1
            events = []
            for l, r in intervals:
                lo = l + 1
                hi = r    # we do -1 at r
                if lo <= hi - 1:
                    # lo <= hi-1  <=> there is some integer y with l < y < r
                    # so active count increases at lo, decreases at hi
                    events.append((lo, 1))
                    events.append((hi, -1))
            events.sort()
            # Sweep to find valid cuts (no interval crosses)
            valid = []
            active = 0
            ei = 0
            E = len(events)
            for y in Y:
                while ei < E and events[ei][0] <= y:
                    active += events[ei][1]
                    ei += 1
                if active == 0:
                    valid.append(y)
            # Need at least two cut positions
            if len(valid) < 2:
                return False
            # Precompute for each valid y the A_count and C_count
            A = []  # number of intervals ending <= y
            C = []  # number of intervals starting >= y
            total = m
            for y in valid:
                a = bisect.bisect_right(ends, y)
                # starts >= y => total - index of first start >= y
                c = total - bisect.bisect_left(starts, y)
                A.append(a)
                C.append(c)
            # Precompute last index in C with C[j] >= 1
            # since C is non-increasing in valid[]
            last_pos = -1
            for idx in range(len(C)-1, -1, -1):
                if C[idx] >= 1:
                    last_pos = idx
                    break
            if last_pos <= 0:
                # either no C[j]>=1 or only at j=0, but we need j>i>=0, so fail
                return False
            # Try to pick i<j such that A[i]>=1, C[j]>=1, and total - A[i] - C[j] >= 1
            # B = total - A - C must be >=1 => C[j] <= total - A[i] - 1
            mval = len(valid)
            for i in range(mval - 1):
                ai = A[i]
                # need at least one in first region
                if ai < 1:
                    continue
                # need room for second and third region
                if ai > total - 2:
                    continue
                # j must be > i and <= last_pos
                if last_pos <= i:
                    break
                # we need C[j] <= total - ai - 1
                limit = total - ai - 1
                # C is non-increasing => C[i+1] is the max in j>i, C[last_pos] is the min >=1
                # if C[last_pos] > limit then even smallest C too big => no j works
                if C[last_pos] > limit:
                    continue
                # find smallest j > i with C[j] <= limit
                lo = i + 1
                hi = last_pos
                # binary search on non-increasing C: we want first index where C[j] <= limit
                while lo < hi:
                    mid = (lo + hi) // 2
                    if C[mid] <= limit:
                        hi = mid
                    else:
                        lo = mid + 1
                # lo is candidate j
                if lo <= last_pos and C[lo] >= 1 and C[lo] <= limit:
                    return True
            return False

        # Build 1D intervals for y and x and test both directions
        y_intervals = [(sy, ey) for _, sy, _, ey in rectangles]
        if check_1d(y_intervals):
            return True
        x_intervals = [(sx, ex) for sx, _, ex, _ in rectangles]
        if check_1d(x_intervals):
            return True
        return False