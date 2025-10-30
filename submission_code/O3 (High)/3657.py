from typing import List

class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        """
        Decide whether two horizontal or two vertical full-length lines can be
        drawn so that the grid is split into three stripes, each stripe
        containing at least one (whole) rectangle and no rectangle is crossed
        by a line.
        """

        def has_three_separated(intervals: List[List[int]]) -> bool:
            """
            intervals – list of [start, end) segments on a line.
            Returns True  ⇔ after joining overlapping segments (but NOT the ones
            that only touch at an endpoint) we obtain at least three disjoint
            segments.  That allows us to put the two cut lines into the two
            gaps between the first-second and the second-third segment.
            """
            intervals.sort()                      # sort by start
            groups = 0
            cur_end = None

            for s, e in intervals:
                if cur_end is None:
                    groups = 1
                    cur_end = e
                    continue

                if s < cur_end:                    # overlaps – extend current
                    if e > cur_end:
                        cur_end = e
                else:                              # real gap (s >= cur_end)
                    groups += 1
                    if groups >= 3:                # already enough segments
                        return True
                    cur_end = e                    # start new segment

            return False                           # < 3 disjoint segments

        # vertical stripes  →   look at x–projections
        x_intervals = [(r[0], r[2]) for r in rectangles]
        if has_three_separated(x_intervals):
            return True

        # horizontal stripes →   look at y–projections
        y_intervals = [(r[1], r[3]) for r in rectangles]
        if has_three_separated(y_intervals):
            return True

        return False