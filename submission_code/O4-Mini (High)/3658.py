from typing import List

class Solution:
    def minDifference(self, nums: List[int]) -> int:
        n = len(nums)
        # Compute the fixed max difference among adjacent known numbers
        fixed_max = 0
        for i in range(n - 1):
            if nums[i] != -1 and nums[i+1] != -1:
                diff = abs(nums[i] - nums[i+1])
                if diff > fixed_max:
                    fixed_max = diff

        # Quick case: if all are missing, we can fill with any equal values => diff = 0
        all_missing = True
        for v in nums:
            if v != -1:
                all_missing = False
                break
        if all_missing:
            return 0

        # Build segments of consecutive -1's with their known neighbors
        # Each segment is represented by a pair (mn, mx)
        #   where mn = min(left_neighbor, right_neighbor)
        #         mx = max(left_neighbor, right_neighbor)
        # For one-sided segments mn == mx == the single neighbor.
        segs = []
        i = 0
        while i < n:
            if nums[i] == -1:
                start = i
                while i < n and nums[i] == -1:
                    i += 1
                # now [start..i-1] was a block of -1
                left_val = nums[start-1] if start > 0 else None
                right_val = nums[i] if i < n else None
                if left_val is None and right_val is None:
                    # whole array missing (already handled) or weird; skip
                    continue
                if left_val is None:
                    mn = mx = right_val
                elif right_val is None:
                    mn = mx = left_val
                else:
                    mn = left_val if left_val < right_val else right_val
                    mx = left_val if left_val > right_val else right_val
                segs.append((mn, mx))
            else:
                i += 1

        # Precompute the minimum D needed so that each segment's interval is non-empty:
        #  interval is [mx - D, mn + D], needs mx - D <= mn + D <=> 2D >= mx - mn
        lower_from_segments = 0
        for mn, mx in segs:
            diff = mx - mn
            # Need D >= ceil(diff/2)
            need = (diff + 1) // 2
            if need > lower_from_segments:
                lower_from_segments = need

        # The answer D must be at least fixed_max and at least lower_from_segments
        lo = max(fixed_max, lower_from_segments)
        hi = 10**9  # safe upper bound for D

        # Sort segments by mn ascending => their "right endpoint" R = mn + D will be ascending too
        segs.sort(key=lambda x: x[0])
        m = len(segs)

        def feasible(D: int) -> bool:
            # If D is too small to make any segment's interval non-empty, fail early
            if D < lower_from_segments:
                return False
            # Greedy hit with at most 2 points
            count = 0
            idx = 0
            while idx < m and count < 2:
                mn, mx = segs[idx]
                # We pick point p at the right end of this interval: R = mn + D
                p = mn + D
                count += 1
                # Skip all segments that p actually covers:
                # coverage condition: p >= L  <=>  p >= (mx - D)
                while idx < m and (mx - D) <= p:
                    # move to next segment
                    idx += 1
                    if idx < m:
                        mn, mx = segs[idx]
                # loop either exhausted or this p stops covering further
            # if we've covered all segments => feasible
            return (idx >= m)

        # Binary search for minimal D in [lo..hi]
        ans = hi
        left, right = lo, hi
        while left <= right:
            mid = (left + right) // 2
            if feasible(mid):
                ans = mid
                right = mid - 1
            else:
                left = mid + 1

        return ans