class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        def can_make_cuts(axis):
            # Collect intervals along the axis
            intervals = []
            for rect in rectangles:
                start = rect[axis]
                end = rect[axis + 2]
                intervals.append((start, end))

            # Sort intervals by start
            intervals.sort()

            # Merge overlapping intervals
            merged_intervals = []
            for interval in intervals:
                if not merged_intervals or merged_intervals[-1][1] < interval[0]:
                    merged_intervals.append(list(interval))
                else:
                    merged_intervals[-1][1] = max(merged_intervals[-1][1], interval[1])

            # Find gaps along the axis where no rectangle exists
            gaps = []
            prev_end = 0
            for interval in merged_intervals:
                if prev_end < interval[0]:
                    gaps.append((prev_end, interval[0]))
                prev_end = interval[1]
            if prev_end < n:
                gaps.append((prev_end, n))

            # Need at least two gaps to make two cuts
            if len(gaps) < 2:
                return False

            # Pick cuts within the first two gaps
            cut1 = (gaps[0][0] + gaps[0][1]) / 2
            cut2 = (gaps[1][0] + gaps[1][1]) / 2

            # Define sections
            sections = [
                (0, cut1),
                (cut1, cut2),
                (cut2, n)
            ]

            # Initialize coverage for each section
            section_coverage = [0, 0, 0]

            for interval in merged_intervals:
                for i, (sec_start, sec_end) in enumerate(sections):
                    # Calculate overlap between interval and section
                    overlap_start = max(interval[0], sec_start)
                    overlap_end = min(interval[1], sec_end)
                    if overlap_start < overlap_end:
                        section_coverage[i] += overlap_end - overlap_start

            # Check if each section has at least some coverage
            if all(coverage > 0 for coverage in section_coverage):
                return True
            else:
                return False

        # Check for vertical cuts (axis=0 for x-axis)
        if can_make_cuts(0):
            return True
        # Check for horizontal cuts (axis=1 for y-axis)
        if can_make_cuts(1):
            return True

        return False