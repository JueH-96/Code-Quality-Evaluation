class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:

        def check_horizontal_cuts(h1, h2, rectangles):
            top = False
            middle = False
            bottom = False
            for r in rectangles:
                if r[3] <= h1:
                    bottom = True
                elif r[1] >= h1 and r[3] <= h2:
                    middle = True
                elif r[1] >= h2:
                    top = True
            return top and middle and bottom

        def check_vertical_cuts(v1, v2, rectangles):
            left = False
            middle = False
            right = False
            for r in rectangles:
                if r[2] <= v1:
                    left = True
                elif r[0] >= v1 and r[2] <= v2:
                    middle = True
                elif r[0] >= v2:
                    right = True
            return left and middle and right

        # Check horizontal cuts
        y_coords = sorted(list(set([r[1] for r in rectangles] + [r[3] for r in rectangles])))
        for i in range(len(y_coords)):
            for j in range(i + 1, len(y_coords)):
                h1 = y_coords[i]
                h2 = y_coords[j]
                if h1 < h2:
                    if check_horizontal_cuts(h1, h2, rectangles):
                        return True

        # Check vertical cuts
        x_coords = sorted(list(set([r[0] for r in rectangles] + [r[2] for r in rectangles])))
        for i in range(len(x_coords)):
            for j in range(i + 1, len(x_coords)):
                v1 = x_coords[i]
                v2 = x_coords[j]
                if v1 < v2:
                    if check_vertical_cuts(v1, v2, rectangles):
                        return True

        return False