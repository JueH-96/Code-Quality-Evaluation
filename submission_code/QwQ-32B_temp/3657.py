import bisect
from typing import List

class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        # Sort rectangles by end_y for processing
        sorted_by_end = sorted(rectangles, key=lambda x: x[3])
        ends_sorted = [rect[3] for rect in sorted_by_end]
        starts_sorted = [rect[1] for rect in sorted_by_end]
        
        m = len(starts_sorted)
        # Compute suffix_min for starts_sorted
        suffix_min = [0] * m
        suffix_min[-1] = starts_sorted[-1]
        for i in range(m-2, -1, -1):
            suffix_min[i] = min(starts_sorted[i], suffix_min[i+1])
        
        # Sort rectangles by start_y for processing
        sorted_by_start = sorted(rectangles, key=lambda x: x[1])
        starts_sorted2 = [rect[1] for rect in sorted_by_start]
        ends_sorted2 = [rect[3] for rect in sorted_by_start]
        
        # Compute prefix_max for ends_sorted2
        prefix_max = [0] * len(ends_sorted2)
        if len(ends_sorted2) > 0:
            prefix_max[0] = ends_sorted2[0]
            for i in range(1, len(ends_sorted2)):
                prefix_max[i] = max(prefix_max[i-1], ends_sorted2[i])
        
        # Collect unique end_y values for possible a candidates
        unique_ends = sorted(list({rect[3] for rect in rectangles}))
        
        # Iterate over each possible a (end_y of some rectangle)
        for a in unique_ends:
            # Find first index where end > a
            i_start = bisect.bisect_right(ends_sorted, a)
            if i_start == 0:
                continue  # No rectangles with end > a
            
            min_start_G = suffix_min[i_start]
            if min_start_G < a:
                continue  # Condition 1 fails
            
            # Determine lower bound for b_candidate
            lower_bound = max(a + 1, min_start_G)
            # Find the first start_y >= lower_bound in starts_sorted2
            j_start = bisect.bisect_left(starts_sorted2, lower_bound)
            
            # Iterate over possible b_candidates starting from j_start
            for j in range(j_start, len(starts_sorted2)):
                b_candidate = starts_sorted2[j]
                # Compute max_end_before(b_candidate)
                k = bisect.bisect_left(starts_sorted2, b_candidate)
                if k == 0:
                    max_end_before = -1
                else:
                    max_end_before = prefix_max[k-1]
                
                if max_end_before > b_candidate:
                    continue  # Condition 2 fails
                
                # Check if middle region has at least one rectangle
                # The minimal end of G's rectangles is ends_sorted[i_start]
                if ends_sorted[i_start] <= b_candidate:
                    return True
        
        return False