class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        def can_cut(intervals):
            # intervals: list of (start, end) for one axis
            intervals.sort()  # sort by start (then end)
            m = len(intervals)
            if m < 3:
                return False
            valid_splits = 0
            max_end = intervals[0][1]
            # try all splits between i and i+1
            for i in range(m - 1):
                if i > 0:
                    max_end = max(max_end, intervals[i][1])
                # a cut between i and i+1 is valid iff
                # no interval in [0..i] crosses the start of i+1
                if max_end <= intervals[i + 1][0]:
                    valid_splits += 1
                    if valid_splits >= 2:
                        return True
            return False

        # Check horizontal cuts (based on y-intervals)
        y_intervals = [(r[1], r[3]) for r in rectangles]
        if can_cut(y_intervals):
            return True
        # Check vertical cuts (based on x-intervals)
        x_intervals = [(r[0], r[2]) for r in rectangles]
        if can_cut(x_intervals):
            return True

        return False