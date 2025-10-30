from typing import List

class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        # Extract the x and y coordinates of the rectangles
        x_coords = []
        y_coords = []

        for rect in rectangles:
            x_coords.append(rect[0])
            x_coords.append(rect[2])
            y_coords.append(rect[1])
            y_coords.append(rect[3])

        # Remove duplicates and sort the coordinates
        x_coords = sorted(set(x_coords))
        y_coords = sorted(set(y_coords))

        # Check for valid vertical cuts
        for i in range(1, len(x_coords) - 1):
            for j in range(i + 1, len(x_coords)):
                left = any(rect[0] < x_coords[i] and rect[2] < x_coords[i] for rect in rectangles)
                middle = any(rect[0] >= x_coords[i] and rect[2] < x_coords[j] for rect in rectangles)
                right = any(rect[0] >= x_coords[j] for rect in rectangles)
                if left and middle and right:
                    return True

        # Check for valid horizontal cuts
        for i in range(1, len(y_coords) - 1):
            for j in range(i + 1, len(y_coords)):
                bottom = any(rect[1] < y_coords[i] and rect[3] < y_coords[i] for rect in rectangles)
                middle = any(rect[1] >= y_coords[i] and rect[3] < y_coords[j] for rect in rectangles)
                top = any(rect[1] >= y_coords[j] for rect in rectangles)
                if bottom and middle and top:
                    return True

        return False