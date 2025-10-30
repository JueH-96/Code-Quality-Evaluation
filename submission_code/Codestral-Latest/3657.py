class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        # Lists to store the x and y coordinates of the rectangle edges
        x_coords = []
        y_coords = []

        # Populate the x and y coordinates lists
        for rect in rectangles:
            x_coords.append(rect[0])
            x_coords.append(rect[2])
            y_coords.append(rect[1])
            y_coords.append(rect[3])

        # Sort the coordinates
        x_coords.sort()
        y_coords.sort()

        # Check for horizontal cuts
        for i in range(1, len(y_coords) - 1):
            if y_coords[i] != y_coords[i - 1] and y_coords[i] != y_coords[i + 1]:
                return True

        # Check for vertical cuts
        for i in range(1, len(x_coords) - 1):
            if x_coords[i] != x_coords[i - 1] and x_coords[i] != x_coords[i + 1]:
                return True

        return False