def solve():
    n = int(input())
    rectangles = []
    for _ in range(n):
        a, b, c, d = map(int, input().split())
        rectangles.append(((a, b), (c, d)))
    
    total_area = 0
    
    for i in range(1, 1 << n):
        current_rectangles_indices = []
        for j in range(n):
            if (i >> j) & 1:
                current_rectangles_indices.append(j)
        
        intersection_x_range = (0, 100)
        intersection_y_range = (0, 100)
        
        first_rect = rectangles[current_rectangles_indices[0]]
        intersection_x_range = (first_rect[0][0], first_rect[0][1])
        intersection_y_range = (first_rect[1][0], first_rect[1][1])
        
        for index in current_rectangles_indices[1:]:
            rect = rectangles[index]
            intersection_x_range = (max(intersection_x_range[0], rect[0][0]), min(intersection_x_range[1], rect[0][1]))
            intersection_y_range = (max(intersection_y_range[0], rect[1][0]), min(intersection_y_range[1], rect[1][1]))
            
        intersection_area = 0
        x_start, x_end = intersection_x_range
        y_start, y_end = intersection_y_range
        
        if x_start < x_end and y_start < y_end:
            intersection_area = (x_end - x_start) * (y_end - y_start)
        elif x_start <= x_end and y_start <= y_end:
            intersection_area = (max(0, x_end - x_start)) * (max(0, y_end - y_start))
        else:
            intersection_area = 0
            
        sign = 1 if len(current_rectangles_indices) % 2 == 1 else -1
        total_area += sign * intersection_area
        
    print(total_area)

if __name__ == '__main__':
    solve()