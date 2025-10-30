class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        x_max = 0
        y_max = 0
        x_max_val = -1
        y_max_val = -1
        n = len(points)
        x_dict = dict()
        y_dict = dict()
        for x, y in points:
            if x in x_dict:
                x_dict[x].append(y)
            else: 
                x_dict[x] = [y]
            if y in y_dict:
                y_dict[y].append(x)
            else: 
                y_dict[y] = [x]
            x_max = max(x, x_max)
            y_max = max(y, y_max)
            x_max_val = max(len(x_dict[x]), x_max_val)
            y_max_val = max(len(y_dict[y]), y_max_val)
        
        x_max_val = min(x_max_val, n-1)
        y_max_val = min(y_max_val, n-1)
        
        result = 2 * (x_max + y_max)
        for x in x_dict:
            new_x = x_max - x
            y_max_inside = max([abs(y - y_dict[y][0]) for y in y_dict])
            result = min(result, 2 * max(new_x, y_max_inside))
        for y in y_dict:
            new_y = y_max - y
            x_max_inside = max([abs(x - x_dict[x][0]) for x in x_dict])
            result = min(result, 2 * max(new_y, x_max_inside))
        
        return result