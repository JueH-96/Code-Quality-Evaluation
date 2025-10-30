class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        """
        We want to see if it is possible to place two horizontal OR two vertical
        cuts so that the grid is split into three sections (top/middle/bottom or
        left/middle/right), each containing at least one rectangle, and no rectangle
        is cut through.

        KEY OBSERVATION:
          • Two rectangles that overlap in (say) the y-dimension cannot be separated
            into different horizontal slices. A horizontal cut spans all x, so if
            two rectangles share any open interval in y, a horizontal line through
            that overlap would slice through at least one of them. Consequently,
            they must lie in the same horizontal "group."
          • Likewise for the x-dimension and vertical cuts.

        Therefore, to decide if we can form three horizontal slices, we look at the
        1D "interval graph" made by projecting each rectangle onto the y-axis:
            interval_i = (start_y, end_y).
        Here, we treat the intervals as half-open [start_y, end_y) so that merely
        "touching" endpoints does not count as an overlap (meaning a cut can be placed
        exactly on that boundary). Two intervals overlap if they have a non-empty
        open intersection. If they do, they must join the same "horizontal group."

        In 1D, the connected components of the "interval overlap graph" can be counted
        by simply sorting the intervals by their start coordinate and merging whenever
        they overlap. The number of connected components in this 1D sense is exactly
        the number of horizontally-separable "blocks."

        If the number of such connected components in y is at least 3, we can place
        two horizontal cuts between these components (since they do not overlap in y,
        there are gaps to place the cuts) and get three horizontal slices.

        If that fails (fewer than 3 blocks in y), we do the same in x. Project each
        rectangle onto x, count how many connected components in that 1D overlap sense.
        If that count is >= 3, we can form three vertical slices. Otherwise, it is not
        possible.

        This procedure takes O(m log m) time (m = number of rectangles) to sort the
        projections and count 1D connected components. It is efficient enough for
        m <= 10^5.

        STEPS:
          1) Build a list of y-intervals from the rectangles: YI = [(sy, ey), ...].
          2) Count how many 1D components in YI using count_components_1D(...).
             If >= 3, return True.
          3) Build a list of x-intervals: XI = [(sx, ex), ...].
          4) Count how many 1D components in XI. If >= 3, return True.
          5) Otherwise, return False.

        EXAMPLES (brief check):
          Example 1 => 3 components in y => True
          Example 2 => 1 component in y, but 3 in x => True
          Example 3 => 2 in y, 2 in x => False
        """

        def count_components_1D(intervals: List[Tuple[int, int]]) -> int:
            """
            Counts how many connected components in the 1D interval-overlap graph,
            treating intervals as half-open [start, end). Intervals overlap if
            they share a non-empty open intersection. We do this by:
            
              • sort by start
              • sweep from left to right, merging all that overlap
              • whenever we find a gap (current interval's start >= the 'running end'),
                we start a new component.
            """
            intervals.sort(key=lambda x: x[0])  # sort by start
            comp_count = 0
            current_end = -1  # something smaller than any valid start

            for s, e in intervals:
                # If this interval starts at or beyond current_end, it's a gap => new component
                if s >= current_end:
                    comp_count += 1
                    current_end = e
                else:
                    # They overlap => merge by extending 'current_end' if needed
                    if e > current_end:
                        current_end = e

            return comp_count

        # Build the y-intervals and x-intervals
        y_intervals = []
        x_intervals = []
        for sx, sy, ex, ey in rectangles:
            # Treat them as half-open for the overlap logic
            # i.e. [sy, ey) and [sx, ex).
            y_intervals.append((sy, ey))
            x_intervals.append((sx, ex))

        # Count how many "blocks" (connected components) in y. If >= 3 => 3 horizontal slices possible
        if count_components_1D(y_intervals) >= 3:
            return True

        # Otherwise, check x
        if count_components_1D(x_intervals) >= 3:
            return True

        return False