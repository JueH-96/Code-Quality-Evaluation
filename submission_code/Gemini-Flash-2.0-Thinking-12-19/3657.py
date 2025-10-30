class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        y_coords = set()
        x_coords = set()
        for rect in rectangles:
            y_coords.add(rect[1])
            y_coords.add(rect[3])
            x_coords.add(rect[0])
            x_coords.add(rect[2])
        
        sorted_y_coords = sorted(list(y_coords))
        sorted_x_coords = sorted(list(x_coords))
        
        for i in range(len(sorted_y_coords)):
            for j in range(len(sorted_y_coords)):
                y1 = sorted_y_coords[i]
                y2 = sorted_y_coords[j]
                if 0 < y1 < y2 < n:
                    section_counts = [0, 0, 0]
                    valid_cut = True
                    rect_sections = [-1] * len(rectangles)
                    for rect_index, rect in enumerate(rectangles):
                        sx, sy, ex, ey = rect
                        section_id = -1
                        if ey <= y1:
                            section_id = 0
                        elif sy >= y1 and ey <= y2:
                            section_id = 1
                        elif sy >= y2:
                            section_id = 2
                        else:
                            valid_cut = False
                            break
                        if section_id != -1:
                            section_counts[section_id] += 1
                            rect_sections[rect_index] = section_id
                        else:
                            valid_cut = False
                            break
                    if not valid_cut:
                        continue
                    if sum(section_counts) == len(rectangles) and all(count > 0 for count in section_counts):
                        return True

        for i in range(len(sorted_x_coords)):
            for j in range(len(sorted_x_coords)):
                x1 = sorted_x_coords[i]
                x2 = sorted_x_coords[j]
                if 0 < x1 < x2 < n:
                    section_counts = [0, 0, 0]
                    valid_cut = True
                    rect_sections = [-1] * len(rectangles)
                    for rect_index, rect in enumerate(rectangles):
                        sx, sy, ex, ey = rect
                        section_id = -1
                        if ex <= x1:
                            section_id = 0
                        elif sx >= x1 and ex <= x2:
                            section_id = 1
                        elif sx >= x2:
                            section_id = 2
                        else:
                            valid_cut = False
                            break
                        if section_id != -1:
                            section_counts[section_id] += 1
                            rect_sections[rect_index] = section_id
                        else:
                            valid_cut = False
                            break
                    if not valid_cut:
                        continue
                    if sum(section_counts) == len(rectangles) and all(count > 0 for count in section_counts):
                        return True
                        
        return False