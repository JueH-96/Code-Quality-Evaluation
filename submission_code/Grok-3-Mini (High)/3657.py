import bisect
from typing import List

class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        def merge_intervals(intervals):
            if not intervals:
                return []
            intervals.sort()  # Sort by start time
            merged = []
            current_start, current_end = intervals[0][0], intervals[0][1]
            for start, end in intervals[1:]:
                if start <= current_end + 1:  # Adjacent or overlapping
                    current_end = max(current_end, end)
                else:
                    merged.append((current_start, current_end))
                    current_start, current_end = start, end
            merged.append((current_start, current_end))
            return merged
        
        def has_valid_cut(L, R, n, F_starts, F_ends):
            C_low = max(L, 1)
            C_high = min(R, n - 1)
            if C_low > C_high:
                return False
            if not F_starts:  # No forbidden intervals
                return True
            # Find the largest i with F_s[i] <= C_low
            idx = bisect.bisect_right(F_starts, C_low) - 1
            if 0 <= idx < len(F_starts) and F_starts[idx] <= C_low and F_ends[idx] >= C_high:
                return False  # No valid cut
            else:
                return True  # There is a valid cut
        
        def check_direction(min_idx, max_idx):
            min_vals = [r[min_idx] for r in rectangles]
            max_vals = [r[max_idx] for r in rectangles]
            min_max_val = min(max_vals)
            max_min_val = max(min_vals)
            if min_max_val >= max_min_val:
                return False
            # Compute forbidden intervals
            forbidden_intervals = []
            for r in rectangles:
                min_val = r[min_idx]
                max_val = r[max_idx]
                if max_val - min_val >= 2:
                    left = min_val + 1
                    right = max_val - 1
                    forbidden_intervals.append((left, right))
            F_intervals = merge_intervals(forbidden_intervals)
            if not F_intervals:
                F_starts = []
                F_ends = []
            else:
                F_starts = [f[0] for f in F_intervals]
                F_ends = [f[1] for f in F_intervals]
            # Find S: rectangles with min_val >= min_max_val and max_val <= max_min_val
            S = []
            for r in rectangles:
                if r[min_idx] >= min_max_val and r[max_idx] <= max_min_val:
                    S.append((r[min_idx], r[max_idx]))  # min and max for r
            if not S:
                return False
            for min_R, max_R in S:
                L1 = min_max_val
                R1 = min_R
                L2 = max_R
                R2 = max_min_val
                if has_valid_cut(L1, R1, n, F_starts, F_ends) and has_valid_cut(L2, R2, n, F_starts, F_ends):
                    return True
            return False
        
        # Check for y-direction (vertical cuts in grid terms, but horizontal cuts in problem)
        if check_direction(1, 3):  # min_y and max_y indices
            return True
        # Check for x-direction (horizontal cuts in grid terms, but vertical cuts in problem)
        if check_direction(0, 2):  # min_x and max_x indices
            return True
        return False