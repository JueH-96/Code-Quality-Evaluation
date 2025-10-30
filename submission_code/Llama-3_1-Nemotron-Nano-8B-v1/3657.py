import bisect
from typing import List

class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        def check_vertical():
            if not rectangles:
                return False
            sorted_rects = sorted(rectangles, key=lambda x: x[0])
            start_x = [r[0] for r in sorted_rects]
            end_x = [r[2] for r in sorted_rects]
            # Precompute prefix_max_end
            prefix_max_end = [0] * (len(sorted_rects) + 1)
            current_max = 0
            for i in range(len(sorted_rects)):
                current_max = max(current_max, end_x[i])
                prefix_max_end[i+1] = current_max
            # Precompute suffix_min_start
            suffix_min_start = [float('inf')] * (len(sorted_rects) + 1)
            current_min = float('inf')
            for i in range(len(sorted_rects)-1, -1, -1):
                current_min = min(current_min, start_x[i])
                suffix_min_start[i] = current_min
            # Sort end_x and start_x
            end_x_sorted = sorted(end_x)
            start_x_sorted = sorted(start_x)
            for L in end_x_sorted:
                # Find R in start_x_sorted > L
                idx_R = bisect.bisect_right(start_x_sorted, L)
                if idx_R >= len(start_x_sorted):
                    continue
                R = start_x_sorted[idx_R]
                # Check max_end_left <= L
                idx_i = bisect.bisect_left(start_x, L)
                max_end_left = prefix_max_end[idx_i]
                if max_end_left > L:
                    continue
                # Check min_start_right >= R
                idx_j = bisect.bisect_right(end_x, R)
                if idx_j < len(sorted_rects):
                    min_start_right = suffix_min_start[idx_j]
                    if min_start_right < R:
                        continue
                else:
                    min_start_right = float('inf')
                # Check there's a rectangle in the middle group
                idx_k = bisect.bisect_left(start_x, L)
                if idx_k >= len(sorted_rects):
                    continue
                # Precompute min_end_from_here for vertical
                min_end_from_here = [0] * (len(sorted_rects) + 1)
                current_min_end = float('inf')
                for i in range(len(sorted_rects)-1, -1, -1):
                    current_min_end = min(current_min_end, end_x[i])
                    min_end_from_here[i] = current_min_end
                min_end_middle = min_end_from_here[idx_k]
                if min_end_middle <= R:
                    return True
            return False

        def check_horizontal():
            if not rectangles:
                return False
            sorted_rects = sorted(rectangles, key=lambda x: x[1])
            start_y = [r[1] for r in sorted_rects]
            end_y = [r[3] for r in sorted_rects]
            # Precompute prefix_max_end
            prefix_max_end = [0] * (len(sorted_rects) + 1)
            current_max = 0
            for i in range(len(sorted_rects)):
                current_max = max(current_max, end_y[i])
                prefix_max_end[i+1] = current_max
            # Precompute suffix_min_start
            suffix_min_start = [float('inf')] * (len(sorted_rects) + 1)
            current_min = float('inf')
            for i in range(len(sorted_rects)-1, -1, -1):
                current_min = min(current_min, start_y[i])
                suffix_min_start[i] = current_min
            # Sort end_y and start_y
            end_y_sorted = sorted(end_y)
            start_y_sorted = sorted(start_y)
            for L in end_y_sorted:
                # Find R in start_y_sorted > L
                idx_R = bisect.bisect_right(start_y_sorted, L)
                if idx_R >= len(start_y_sorted):
                    continue
                R = start_y_sorted[idx_R]
                # Check max_end_left <= L
                idx_i = bisect.bisect_left(start_y, L)
                max_end_left = prefix_max_end[idx_i]
                if max_end_left > L:
                    continue
                # Check min_start_right >= R
                idx_j = bisect.bisect_right(end_y, R)
                if idx_j < len(sorted_rects):
                    min_start_right = suffix_min_start[idx_j]
                    if min_start_right < R:
                        continue
                else:
                    min_start_right = float('inf')
                # Check there's a rectangle in the middle group
                idx_k = bisect.bisect_left(start_y, L)
                if idx_k >= len(sorted_rects):
                    continue
                # Precompute min_end_from_here for horizontal
                min_end_from_here = [0] * (len(sorted_rects) + 1)
                current_min_end = float('inf')
                for i in range(len(sorted_rects)-1, -1, -1):
                    current_min_end = min(current_min_end, end_y[i])
                    min_end_from_here[i] = current_min_end
                min_end_middle = min_end_from_here[idx_k]
                if min_end_middle <= R:
                    return True
            return False

        return check_vertical() or check_horizontal()