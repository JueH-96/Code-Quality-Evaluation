import collections

class Solution:
    def separateSquares(self, squares: List[List[int]]) -> float:
        total_area = 0
        events = collections.defaultdict(int)
        for sq in squares:
            x, y, l = sq
            total_area += l * l
            events[y] += l
            events[y + l] -= l
        
        target = total_area / 2.0
        event_ys = sorted(events.keys())
        current_area = 0.0
        current_slope = 0.0
        last_y = 0.0
        
        for y in event_ys:
            seg_length = y - last_y
            seg_area = current_slope * seg_length
            if current_area < target:
                if current_area + seg_area >= target:
                    return last_y + (target - current_area) / current_slope
                current_area += seg_area
            if abs(current_area - target) < 1e-7:
                return y
            current_slope += events[y]
            last_y = y