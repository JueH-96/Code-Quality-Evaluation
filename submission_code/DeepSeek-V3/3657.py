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
        # The cuts should divide the rectangles into three sections
        # Each section should have at least one rectangle
        # So, we need to find two y positions such that:
        # - The first cut is above at least one rectangle's bottom
        # - The second cut is below at least one rectangle's top
        # - The middle section has at least one rectangle
        
        # To find such y positions, we can iterate through all possible pairs of y coordinates
        # and check if they satisfy the conditions
        
        # Since the number of y coordinates is small (up to 2 * rectangles.length), we can afford to try all pairs
        for i in range(len(sorted_y)):
            for j in range(i+1, len(sorted_y)):
                y1 = sorted_y[i]
                y2 = sorted_y[j]
                if y1 >= y2:
                    continue
                # Now, check if the cuts at y1 and y2 divide the rectangles into three sections
                # Each section should have at least one rectangle
                # We can count the number of rectangles in each section
                # Section 1: y < y1
                # Section 2: y1 <= y < y2
                # Section 3: y >= y2
                count1 = 0
                count2 = 0
                count3 = 0
                for rect in rectangles:
                    bottom = rect[1]
                    top = rect[3]
                    if top <= y1:
                        count1 += 1
                    elif bottom >= y2:
                        count3 += 1
                    else:
                        count2 += 1
                if count1 > 0 and count2 > 0 and count3 > 0:
                    return True
        
        # Try vertical cuts
        # Similar logic as above
        for i in range(len(sorted_x)):
            for j in range(i+1, len(sorted_x)):
                x1 = sorted_x[i]
                x2 = sorted_x[j]
                if x1 >= x2:
                    continue
                # Now, check if the cuts at x1 and x2 divide the rectangles into three sections
                # Each section should have at least one rectangle
                # Section 1: x < x1
                # Section 2: x1 <= x < x2
                # Section 3: x >= x2
                count1 = 0
                count2 = 0
                count3 = 0
                for rect in rectangles:
                    left = rect[0]
                    right = rect[2]
                    if right <= x1:
                        count1 += 1
                    elif left >= x2:
                        count3 += 1
                    else:
                        count2 += 1
                if count1 > 0 and count2 > 0 and count3 > 0:
                    return True
        
        # If no valid cuts found
        return False