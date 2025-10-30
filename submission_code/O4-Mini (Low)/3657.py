from typing import List

class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        # Helper to check along one axis: intervals = list of [start, end]
        def can_split(intervals: List[List[int]]) -> bool:
            m = len(intervals)
            if m < 3:
                return False

            # Sort by end coordinate
            intervals.sort(key=lambda x: x[1])
            starts = [iv[0] for iv in intervals]
            ends = [iv[1] for iv in intervals]

            # Build suffix_min_start: at i, min start among intervals[i:]
            INF = 10**18
            suffix_min_start = [INF] * (m + 1)
            for i in range(m - 1, -1, -1):
                suffix_min_start[i] = min(starts[i], suffix_min_start[i + 1])

            # validSecond[j] = True if we can place second cut after j
            # i.e., intervals[j].end <= cut2, and all intervals[j+1:] start >= cut2
            validSecond = [False] * m
            # only j up to m-2 can leave a non-empty third group
            for j in range(m - 1):
                # cut2 = ends[j]
                if suffix_min_start[j + 1] >= ends[j]:
                    validSecond[j] = True

            # nextValid[i] = True if there's some validSecond[j] for j >= i
            nextValid = [False] * (m + 1)
            for i in range(m - 1, -1, -1):
                nextValid[i] = validSecond[i] or nextValid[i + 1]

            # Try placing first cut after i (i from 0..m-3 to leave two groups)
            for i in range(0, m - 2):
                # cut1 = ends[i]
                # need all intervals[i+1:] start >= cut1
                if suffix_min_start[i + 1] < ends[i]:
                    continue
                # need a valid second cut in intervals[i+1..]
                # that is nextValid at position i+1, and the chosen j <= m-2
                if nextValid[i + 1]:
                    return True

            return False

        # Build horizontal intervals [start_y, end_y]
        y_intervals = [[r[1], r[3]] for r in rectangles]
        # Build vertical intervals [start_x, end_x]
        x_intervals = [[r[0], r[2]] for r in rectangles]

        # Check horizontal or vertical possibility
        return can_split(y_intervals) or can_split(x_intervals)