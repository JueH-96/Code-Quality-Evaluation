class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        min_x = float('inf')
        max_x = float('-inf')
        min_y = float('inf')
        max_y = float('-inf')
        x_coords = set()
        y_coords = set()

        for rect in rectangles:
            min_x = min(min_x, rect[0])
            max_x = max(max_x, rect[2])
            min_y = min(min_y, rect[1])
            max_y = max(max_y, rect[3])
            x_coords.add(rect[0])
            x_coords.add(rect[2])
            y_coords.add(rect[1])
            y_coords.add(rect[3])

        x_coords = sorted(list(x_coords))
        y_coords = sorted(list(y_coords))

        def check_cuts(coords, cuts):
            if len(coords) < 3 : return False
            
            for i in range(1, len(coords) -1):
                for j in range(i+1, len(coords)):
                    
                    count1 = 0
                    count2 = 0
                    count3 = 0
                    
                    for rect in rectangles:
                        if (coords[i-1] <= rect[0] < coords[i] and coords[i-1] <= rect[2] < coords[i]) or \
                           (coords[i-1] <= rect[0] < coords[i] and coords[i] <= rect[2]) or \
                           (coords[i] <= rect[0] and coords[i-1] <= rect[2] < coords[i]) :
                            count1 +=1
                        elif (coords[i] <= rect[0] < coords[j] and coords[i] <= rect[2] < coords[j]) or \
                             (coords[i] <= rect[0] < coords[j] and coords[j] <= rect[2]) or \
                             (coords[j] <= rect[0] and coords[i] <= rect[2] < coords[j]):
                            count2 +=1
                        elif coords[j] <= rect[0] and coords[j] <= rect[2]:
                            count3 +=1
                    if count1 > 0 and count2 > 0 and count3 > 0:
                        return True
            return False

        return check_cuts(x_coords,2) or check_cuts(y_coords,2)