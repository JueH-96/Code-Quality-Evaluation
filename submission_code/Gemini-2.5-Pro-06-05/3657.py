from typing import List

class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        """
        Determines if two horizontal or two vertical cuts can be made to partition
        the grid and rectangles into three non-empty sections.
        """
        
        def can_partition(intervals: List[List[int]]) -> bool:
            """
            Checks if a list of intervals can be partitioned into at least 3
            non-overlapping groups. This is equivalent to finding the number of
            connected components in the corresponding interval graph.
            """
            # Sort intervals by their start points.
            intervals.sort(key=lambda x: x[0])
            
            num_components = 0
            i = 0
            k = len(intervals)
            
            while i < k:
                # Each iteration of this outer loop finds one connected component.
                num_components += 1
                
                # The rightmost boundary of the current merged component.
                max_end = intervals[i][1]
                
                # Greedily merge all subsequent intervals that overlap with the current component.
                # An interval [s_j, e_j) overlaps with the current merged component
                # if its start s_j is less than the component's max_end.
                j = i + 1
                while j < k and intervals[j][0] < max_end:
                    max_end = max(max_end, intervals[j][1])
                    j += 1
                
                # The component consists of intervals from the original index i to j-1.
                # The next potential component starts at index j.
                i = j
            
            return num_components >= 3

        # Check for the possibility of two horizontal cuts.
        # This requires partitioning the rectangles based on their y-intervals.
        y_intervals = [[r[1], r[3]] for r in rectangles]
        if can_partition(y_intervals):
            return True
            
        # Check for the possibility of two vertical cuts.
        # This requires partitioning the rectangles based on their x-intervals.
        x_intervals = [[r[0], r[2]] for r in rectangles]
        if can_partition(x_intervals):
            return True
            
        return False