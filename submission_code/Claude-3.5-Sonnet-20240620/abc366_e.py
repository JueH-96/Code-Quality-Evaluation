# YOUR CODE HERE
from collections import defaultdict

def manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def count_points(N, D, points):
    x_coords = sorted(set(x for x, _ in points))
    y_coords = sorted(set(y for _, y in points))
    
    x_index = {x: i for i, x in enumerate(x_coords)}
    y_index = {y: i for i, y in enumerate(y_coords)}
    
    grid = defaultdict(int)
    for x, y in points:
        grid[(x_index[x], y_index[y])] = 1
    
    prefix_sum = defaultdict(int)
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            prefix_sum[(i, j)] = (
                prefix_sum[(i-1, j)] + 
                prefix_sum[(i, j-1)] - 
                prefix_sum[(i-1, j-1)] + 
                grid[(i, j)]
            )
    
    count = 0
    for x in range(min(x_coords) - D, max(x_coords) + D + 1):
        for y in range(min(y_coords) - D, max(y_coords) + D + 1):
            total_distance = 0
            x_left = x_right = x
            y_bottom = y_top = y
            
            while x_left > min(x_coords) or x_right < max(x_coords) or y_bottom > min(y_coords) or y_top < max(y_coords):
                if x_left > min(x_coords):
                    x_left -= 1
                if x_right < max(x_coords):
                    x_right += 1
                if y_bottom > min(y_coords):
                    y_bottom -= 1
                if y_top < max(y_coords):
                    y_top += 1
                
                i_left = x_index[x_left] if x_left in x_index else -1
                i_right = x_index[x_right] if x_right in x_index else len(x_coords)
                j_bottom = y_index[y_bottom] if y_bottom in y_index else -1
                j_top = y_index[y_top] if y_top in y_index else len(y_coords)
                
                points_in_range = (
                    prefix_sum[(i_right, j_top)] - 
                    prefix_sum[(i_left, j_top)] - 
                    prefix_sum[(i_right, j_bottom)] + 
                    prefix_sum[(i_left, j_bottom)]
                )
                
                total_distance += points_in_range * (
                    manhattan_distance(x, y, x_left, y_bottom) - 
                    manhattan_distance(x, y, x_left-1, y_bottom-1)
                )
                
                if total_distance > D:
                    break
            
            if total_distance <= D:
                count += 1
    
    return count

N, D = map(int, input().split())
points = [tuple(map(int, input().split())) for _ in range(N)]

result = count_points(N, D, points)
print(result)