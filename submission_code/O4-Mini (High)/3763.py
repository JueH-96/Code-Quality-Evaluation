from typing import List

class Solution:
    def separateSquares(self, squares: List[List[int]]) -> float:
        # Build sweep‐line events: at y=start, slope +l; at y=end, slope -l
        events = []
        total_area = 0
        for x, y, l in squares:
            events.append((y,     l))
            events.append((y + l, -l))
            total_area += l * l

        half = total_area * 0.5
        # Sort by y‐coordinate
        events.sort(key=lambda e: e[0])

        f_prev = 0.0     # accumulated area up to prev_y
        slope = 0.0      # current derivative (sum of active side‐lengths)
        prev_y = events[0][0]
        i = 0
        n = len(events)

        while i < n:
            y = events[i][0]
            dy = y - prev_y
            # Over the interval [prev_y, y), area grows by slope * dy
            if dy > 0 and slope > 0:
                # Check if the half‐area is reached in this segment
                if f_prev + slope * dy >= half:
                    return prev_y + (half - f_prev) / slope
                f_prev += slope * dy

            # Advance the baseline to y
            prev_y = y
            # Process all events at this y
            while i < n and events[i][0] == y:
                slope += events[i][1]
                i += 1

        # In theory we should never get here before finding the split.
        return float(prev_y)