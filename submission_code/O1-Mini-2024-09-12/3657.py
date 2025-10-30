from typing import List

class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        def can_make_two_cuts(rects, axis_start, axis_end):
            # Sort rectangles based on start axis
            rects_sorted = sorted(rects, key=lambda x: x[axis_start])
            prefix_max_end = []
            current_max = -1
            for rect in rects_sorted:
                current_max = max(current_max, rect[axis_end])
                prefix_max_end.append(current_max)
            
            # Sort rectangles based on end axis
            rects_sorted_end = sorted(rects, key=lambda x: x[axis_end])
            suffix_min_start = [0] * len(rects)
            current_min = float('inf')
            for i in range(len(rects_sorted_end)-1, -1, -1):
                current_min = min(current_min, rects_sorted_end[i][axis_start])
                suffix_min_start[i] = current_min
            
            # Iterate to find two cuts
            # First cut after i, second cut after j
            for i in range(len(prefix_max_end) - 2):
                y1 = prefix_max_end[i]
                # Binary search for the second cut y2 starting from i+1
                left = i + 1
                right = len(prefix_max_end) - 1
                while left <= right:
                    mid = (left + right) // 2
                    y2 = prefix_max_end[mid]
                    if y2 >= y1 and y2 < suffix_min_start[mid + 1]:
                        # Ensure all rectangles up to mid have end <= y2
                        # and all rectangles after mid have start >= y2
                        # Also, check that there are at least one rectangle in each group
                        group1_count = i + 1
                        group2_count = mid - i
                        group3_count = len(rects) - (mid + 1)
                        if group1_count >=1 and group2_count >=1 and group3_count >=1:
                            return True
                        else:
                            left = mid + 1
                    elif y2 < suffix_min_start[mid + 1]:
                        right = mid - 1
                    else:
                        left = mid + 1
            return False

        # Check for horizontal cuts (y-axis)
        horizontal_possible = can_make_two_cuts(rectangles, 1, 3)
        
        if horizontal_possible:
            return True
        
        # Check for vertical cuts (x-axis)
        vertical_possible = can_make_two_cuts(rectangles, 0, 2)
        
        return vertical_possible