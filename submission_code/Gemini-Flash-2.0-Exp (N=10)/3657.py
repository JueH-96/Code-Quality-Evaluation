class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        def check_horizontal_cuts(cuts):
            if len(cuts) != 2:
                return False
            
            cuts.sort()
            if cuts[0] <= 0 or cuts[0] >= n or cuts[1] <= 0 or cuts[1] >= n or cuts[0] >= cuts[1]:
                return False

            sections = [[] for _ in range(3)]
            for rect in rectangles:
                start_x, start_y, end_x, end_y = rect
                if end_y <= cuts[0]:
                    sections[0].append(rect)
                elif start_y >= cuts[1]:
                    sections[2].append(rect)
                elif start_y < cuts[1] and end_y > cuts[0]:
                    sections[1].append(rect)
                else:
                    return False
            
            return all(len(section) > 0 for section in sections)

        def check_vertical_cuts(cuts):
            if len(cuts) != 2:
                return False
            
            cuts.sort()
            if cuts[0] <= 0 or cuts[0] >= n or cuts[1] <= 0 or cuts[1] >= n or cuts[0] >= cuts[1]:
                return False

            sections = [[] for _ in range(3)]
            for rect in rectangles:
                start_x, start_y, end_x, end_y = rect
                if end_x <= cuts[0]:
                    sections[0].append(rect)
                elif start_x >= cuts[1]:
                    sections[2].append(rect)
                elif start_x < cuts[1] and end_x > cuts[0]:
                    sections[1].append(rect)
                else:
                    return False
            
            return all(len(section) > 0 for section in sections)

        
        
        
        ys = set()
        xs = set()
        for rect in rectangles:
            ys.add(rect[1])
            ys.add(rect[3])
            xs.add(rect[0])
            xs.add(rect[2])
        
        ys = sorted(list(ys))
        xs = sorted(list(xs))
        
        
        for i in range(len(ys)):
            for j in range(i+1, len(ys)):
                if check_horizontal_cuts([ys[i], ys[j]]):
                    return True
        
        for i in range(len(xs)):
            for j in range(i+1, len(xs)):
                if check_vertical_cuts([xs[i], xs[j]]):
                    return True
        
        return False