class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        def check_horizontal_cuts(y1, y2):
            sections = [[] for _ in range(3)]
            for rect in rectangles:
                if rect[3] <= y1:
                    sections[0].append(rect)
                elif rect[1] >= y2:
                    sections[2].append(rect)
                elif rect[1] < y1 and rect[3] > y2:
                    return False
                elif rect[3] > y1 and rect[1] < y1:
                    return False
                elif rect[3] > y2 and rect[1] < y2:
                    return False
                else:
                    sections[1].append(rect)
            return all(len(section) > 0 for section in sections)

        def check_vertical_cuts(x1, x2):
            sections = [[] for _ in range(3)]
            for rect in rectangles:
                if rect[2] <= x1:
                    sections[0].append(rect)
                elif rect[0] >= x2:
                    sections[2].append(rect)
                elif rect[0] < x1 and rect[2] > x2:
                    return False
                elif rect[2] > x1 and rect[0] < x1:
                    return False
                elif rect[2] > x2 and rect[0] < x2:
                    return False
                else:
                    sections[1].append(rect)
            return all(len(section) > 0 for section in sections)

        for i in range(1, n):
            for j in range(i + 1, n):
                if check_horizontal_cuts(i, j):
                    return True
                if check_vertical_cuts(i, j):
                    return True

        return False