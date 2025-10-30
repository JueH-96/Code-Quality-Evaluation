class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        def check_horizontal(rectangles):
            y_coords = set()
            for _, y1, _, y2 in rectangles:
                y_coords.add(y1)
                y_coords.add(y2)
            y_coords = sorted(list(y_coords))
            if len(y_coords) < 4:
                return False
            
            for i in range(1, len(y_coords) - 1):
                for j in range(i + 1, len(y_coords) - 1):
                    y1_cut = y_coords[i]
                    y2_cut = y_coords[j]
                    
                    group1 = []
                    group2 = []
                    group3 = []
                    
                    valid = True
                    for x1, y1, x2, y2 in rectangles:
                        if y2 <= y1_cut:
                            group1.append((x1, y1, x2, y2))
                        elif y1 >= y2_cut:
                            group3.append((x1, y1, x2, y2))
                        elif y1 < y1_cut and y2 > y1_cut and y2 <= y2_cut:
                            group2.append((x1, y1, x2, y2))
                        elif y1 < y2_cut and y2 > y2_cut:
                            group3.append((x1,y1, x2, y2))
                        elif y1 >= y1_cut and y2 <= y2_cut:
                            group2.append((x1, y1, x2, y2))
                        else:
                            valid = False
                            break
                    if valid and group1 and group2 and group3:
                        return True
            return False

        def check_vertical(rectangles):
            x_coords = set()
            for x1, _, x2, _ in rectangles:
                x_coords.add(x1)
                x_coords.add(x2)
            x_coords = sorted(list(x_coords))
            if len(x_coords) < 4:
                return False

            for i in range(1, len(x_coords) - 1):
                for j in range(i + 1, len(x_coords) - 1):
                    x1_cut = x_coords[i]
                    x2_cut = x_coords[j]

                    group1 = []
                    group2 = []
                    group3 = []

                    valid = True
                    for x1, y1, x2, y2 in rectangles:
                        if x2 <= x1_cut:
                            group1.append((x1, y1, x2, y2))
                        elif x1 >= x2_cut:
                            group3.append((x1, y1, x2, y2))
                        elif x1 < x1_cut and x2 > x1_cut and x2 <= x2_cut:
                            group2.append((x1, y1, x2, y2))
                        elif x1 < x2_cut and x2 > x2_cut:
                            group3.append((x1, y1, x2, y2))
                        elif x1 >= x1_cut and x2 <= x2_cut:
                            group2.append((x1, y1, x2, y2))
                        
                        else:
                            valid = False
                            break
                    if valid and group1 and group2 and group3:
                        return True
            return False

        return check_horizontal(rectangles) or check_vertical(rectangles)