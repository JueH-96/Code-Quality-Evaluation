from typing import List

class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        # Extract all x and y coordinates
        x_coords = set()
        y_coords = set()
        for rect in rectangles:
            x_coords.add(rect[0])
            x_coords.add(rect[2])
            y_coords.add(rect[1])
            y_coords.add(rect[3])
        
        # Sort the coordinates
        sorted_x = sorted(x_coords)
        sorted_y = sorted(y_coords)
        
        # Try horizontal cuts
        # We need to find two y positions to cut
        # The cuts should divide the rectangles into three non-empty groups
        # So, we need to find two y positions such that:
        # - The first cut is above at least one rectangle's bottom
        # - The second cut is below at least one rectangle's top
        # - The middle section has at least one rectangle
        # So, we need to find two y positions y1 and y2 (y1 < y2) such that:
        # - There exists a rectangle with bottom < y1
        # - There exists a rectangle with top > y2
        # - There exists a rectangle with bottom >= y1 and top <= y2
        
        # To find such y1 and y2, we can iterate through all possible pairs in sorted_y
        # Since the rectangles are non-overlapping, the sorted_y list will have at least 2 elements
        # So, we can try all possible pairs of y1 and y2 where y1 < y2
        
        # Iterate through all possible pairs of y1 and y2
        for i in range(len(sorted_y) - 1):
            y1 = sorted_y[i]
            for j in range(i + 1, len(sorted_y)):
                y2 = sorted_y[j]
                # Check if there is at least one rectangle below y1
                has_below = False
                # Check if there is at least one rectangle above y2
                has_above = False
                # Check if there is at least one rectangle between y1 and y2
                has_middle = False
                for rect in rectangles:
                    if rect[1] < y1:
                        has_below = True
                    if rect[3] > y2:
                        has_above = True
                    if rect[1] >= y1 and rect[3] <= y2:
                        has_middle = True
                if has_below and has_above and has_middle:
                    return True
        
        # Try vertical cuts
        # Similar logic as horizontal cuts
        for i in range(len(sorted_x) - 1):
            x1 = sorted_x[i]
            for j in range(i + 1, len(sorted_x)):
                x2 = sorted_x[j]
                # Check if there is at least one rectangle to the left of x1
                has_left = False
                # Check if there is at least one rectangle to the right of x2
                has_right = False
                # Check if there is at least one rectangle between x1 and x2
                has_middle = False
                for rect in rectangles:
                    if rect[0] < x1:
                        has_left = True
                    if rect[2] > x2:
                        has_right = True
                    if rect[0] >= x1 and rect[2] <= x2:
                        has_middle = True
                if has_left and has_right and has_middle:
                    return True
        
        # If no valid cuts found
        return False