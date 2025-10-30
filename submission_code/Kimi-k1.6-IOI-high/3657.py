import bisect
from typing import List

class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        # Check for valid horizontal cuts using start_y and end_y
        horizontal = self.check_cuts(rectangles, start_index=1, end_index=3)
        if horizontal:
            return True
        
        # Check for valid vertical cuts using start_x and end_x
        vertical = self.check_cuts(rectangles, start_index=0, end_index=2)
        if vertical:
            return True
        
        return False
    
    def check_cuts(self, rectangles, start_index, end_index):
        # Extract all start coordinates and sort them
        starts = sorted([rect[start_index] for rect in rectangles])
        # Extract all end coordinates and sort them
        ends_all = sorted([rect[end_index] for rect in rectangles])
        # Get unique end coordinates and sort them
        ends_unique = sorted(set(rect[end_index] for rect in rectangles))
        
        # Sort the rectangles by the start coordinate to facilitate checks
        sorted_rects = sorted(rectangles, key=lambda x: x[start_index])
        # Extract the sorted start coordinates for binary search
        sorted_starts = [r[start_index] for r in sorted_rects]
        # Prepare a suffix array to track the minimal end coordinate from each position to the end
        suffix_min_end = [0] * len(sorted_rects)
        if sorted_rects:
            # Initialize the last element of the suffix array
            suffix_min_end[-1] = sorted_rects[-1][end_index]
            # Populate the suffix array from right to left
            for i in range(len(sorted_rects) - 2, -1, -1):
                suffix_min_end[i] = min(sorted_rects[i][end_index], suffix_min_end[i + 1])
        
        # Check each unique end coordinate as a potential lower cut position
        for e in ends_unique:
            # Find the position in starts where elements are greater than e
            pos = bisect.bisect_right(starts, e)
            if pos >= len(starts):
                continue  # No valid start coordinate found beyond e
            
            s_candidate = starts[pos]
            # Check if there are enough rectangles ending at or before e
            cnt_end_le_e = bisect.bisect_right(ends_all, e)
            if cnt_end_le_e == 0:
                continue
            # Check if there are enough rectangles starting at or after s_candidate
            cnt_start_ge_s = len(starts) - bisect.bisect_left(starts, s_candidate)
            if cnt_start_ge_s == 0:
                continue
            # Find the first rectangle with start >= e in the sorted list
            left = bisect.bisect_left(sorted_starts, e)
            if left >= len(sorted_rects):
                continue  # No such rectangle exists
            # Check if the minimal end in the remaining rectangles is <= s_candidate
            min_end = suffix_min_end[left]
            if min_end <= s_candidate:
                return True  # Valid cuts found
        
        return False