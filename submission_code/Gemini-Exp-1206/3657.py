class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        def check_horizontal_cuts(rectangles):
            y_coords = set()
            for rect in rectangles:
                y_coords.add(rect[1])
                y_coords.add(rect[3])
            y_coords = sorted(list(y_coords))
            
            for i in range(len(y_coords) - 2):
                for j in range(i + 1, len(y_coords) - 1):
                    cut1 = y_coords[i]
                    cut2 = y_coords[j]
                    
                    section1 = []
                    section2 = []
                    section3 = []
                    
                    for rect in rectangles:
                        if rect[3] <= cut1:
                            section1.append(rect)
                        elif rect[1] >= cut2:
                            section3.append(rect)
                        elif rect[1] >= cut1 and rect[3] <= cut2:
                            section2.append(rect)
                    
                    if len(section1) > 0 and len(section2) > 0 and len(section3) > 0:
                        return True
            return False

        def check_vertical_cuts(rectangles):
            x_coords = set()
            for rect in rectangles:
                x_coords.add(rect[0])
                x_coords.add(rect[2])
            x_coords = sorted(list(x_coords))

            for i in range(len(x_coords) - 2):
                for j in range(i + 1, len(x_coords) - 1):
                    cut1 = x_coords[i]
                    cut2 = x_coords[j]

                    section1 = []
                    section2 = []
                    section3 = []

                    for rect in rectangles:
                        if rect[2] <= cut1:
                            section1.append(rect)
                        elif rect[0] >= cut2:
                            section3.append(rect)
                        elif rect[0] >= cut1 and rect[2] <= cut2:
                            section2.append(rect)

                    if len(section1) > 0 and len(section2) > 0 and len(section3) > 0:
                        return True
            return False

        return check_horizontal_cuts(rectangles) or check_vertical_cuts(rectangles)