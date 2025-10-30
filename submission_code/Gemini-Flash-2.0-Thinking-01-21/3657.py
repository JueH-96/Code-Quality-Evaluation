class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        def is_valid_horizontal_cut(y1, y2, rects):
            section_counts = [0, 0, 0]
            for rect in rects:
                end_y = rect[3]
                if end_y <= y1:
                    section_counts[0] += 1
                elif end_y <= y2:
                    section_counts[1] += 1
                else:
                    section_counts[2] += 1
            return all(count > 0 for count in section_counts)
        
        def is_valid_vertical_cut(x1, x2, rects):
            section_counts = [0, 0, 0]
            for rect in rects:
                end_x = rect[2]
                if end_x <= x1:
                    section_counts[0] += 1
                elif end_x <= x2:
                    section_counts[1] += 1
                else:
                    section_counts[2] += 1
            return all(count > 0 for count in section_counts)
            
        for y1 in range(1, n):
            for y2 in range(y1 + 1, n):
                if is_valid_horizontal_cut(y1, y2, rectangles):
                    return True
                    
        for x1 in range(1, n):
            for x2 in range(x1 + 1, n):
                if is_valid_vertical_cut(x1, x2, rectangles):
                    return True
                    
        return False