import bisect
from typing import List

class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        # Check vertical cuts possibility
        vertical_possible = False
        end_x = [r[2] for r in rectangles]
        start_x = [r[0] for r in rectangles]
        sorted_end_x = sorted(end_x)
        sorted_start_x = sorted(start_x)
        
        for s, _, e, _ in rectangles:
            left_count = bisect.bisect_right(sorted_end_x, s)
            if left_count >= 1:
                right_count = len(sorted_start_x) - bisect.bisect_left(sorted_start_x, e)
                if right_count >= 1:
                    vertical_possible = True
                    break
        
        # Check horizontal cuts possibility
        horizontal_possible = False
        end_y = [r[3] for r in rectangles]
        start_y = [r[1] for r in rectangles]
        sorted_end_y = sorted(end_y)
        sorted_start_y = sorted(start_y)
        
        for _, s, _, e in rectangles:
            left_count = bisect.bisect_right(sorted_end_y, s)
            if left_count >= 1:
                right_count = len(sorted_start_y) - bisect.bisect_left(sorted_start_y, e)
                if right_count >= 1:
                    horizontal_possible = True
                    break
        
        return vertical_possible or horizontal_possible