def solve():
    n, d = map(int, input().split())
    points = []
    for _ in range(n):
        points.append(tuple(map(int, input().split())))
    
    def calculate_g(x, points):
        g_val = 0
        for px, py in points:
            g_val += abs(x - px)
        return g_val
        
    def calculate_h(y, points):
        h_val = 0
        for px, py in points:
            h_val += abs(y - py)
        return h_val
        
    count = 0
    
    min_x = min(p[0] for p in points) if points else 0
    max_x = max(p[0] for p in points) if points else 0
    min_y = min(p[1] for p in points) if points else 0
    max_y = max(p[1] for p in points) if points else 0
    
    x_range_start = min_x - d - 2
    x_range_end = max_x + d + 2
    y_range_start = min_y - d - 2
    y_range_end = max_y + d + 2

    for x in range(x_range_start, x_range_end + 1):
        g_x = calculate_g(x, points)
        if g_x <= d:
            d_x = d - g_x
            y_start_found = False
            y_start = -float('inf')
            low_y = y_range_start
            high_y = y_range_end
            while low_y <= high_y:
                mid_y = (low_y + high_y) // 2
                h_y = calculate_h(mid_y, points)
                if h_y <= d_x:
                    y_start = mid_y
                    y_start_found = True
                    high_y = mid_y - 1
                else:
                    low_y = mid_y + 1
                    
            y_end_found = False
            y_end = float('inf')
            low_y = y_range_start
            high_y = y_range_end
            while low_y <= high_y:
                mid_y = (low_y + high_y) // 2
                h_y = calculate_h(mid_y, points)
                if h_y <= d_x:
                    y_end = mid_y
                    y_end_found = True
                    low_y = mid_y + 1
                else:
                    high_y = mid_y - 1
                    
            if y_start_found and y_end_found:
                if y_start <= y_end:
                    count += (y_end - y_start + 1)

    print(count)

if __name__ == '__main__':
    solve()