def solve():
    n, d = map(int, input().split())
    points = []
    for _ in range(n):
        points.append(tuple(map(int, input().split())))
    
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    def get_g(x):
        value = 0
        for xi in x_coords:
            value += abs(x - xi)
        return value
        
    def get_h(y):
        value = 0
        for yi in y_coords:
            value += abs(y - yi)
        return value
        
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)
    
    x_start_range = min_x - d - 50
    x_end_range = max_x + d + 50
    y_start_range = min_y - d - 50
    y_end_range = max_y + d + 50
    
    count = 0
    
    x_values_to_check = []
    
    x_median_index = (n + 1) // 2 - 1
    sorted_x_coords = sorted(x_coords)
    x_median = sorted_x_coords[x_median_index] if n > 0 else 0
    
    x_lower_bound = None
    x_upper_bound = None
    
    current_x = x_median
    while True:
        if get_g(current_x) <= d:
            x_lower_bound = current_x
            current_x -= 1
        else:
            break
    if x_lower_bound is None:
        x_lower_bound = x_median
        
    current_x = x_median
    while True:
        if get_g(current_x) <= d:
            x_upper_bound = current_x
            current_x += 1
        else:
            break
    if x_upper_bound is None:
        x_upper_bound = x_median

    if x_lower_bound is None or x_upper_bound is None:
        print(0)
        return
        
    x_range_start = x_lower_bound
    x_range_end = x_upper_bound
    
    for x in range(x_range_start, x_range_end + 1):
        g_x = get_g(x)
        if g_x <= d:
            d_prime = d - g_x
            y_median_index = (n + 1) // 2 - 1
            sorted_y_coords = sorted(y_coords)
            y_median = sorted_y_coords[y_median_index] if n > 0 else 0
            
            y_lower_bound_y = None
            y_upper_bound_y = None
            
            current_y = y_median
            while True:
                if get_h(current_y) <= d_prime:
                    y_lower_bound_y = current_y
                    current_y -= 1
                else:
                    break
            if y_lower_bound_y is None:
                y_lower_bound_y = y_median
                
            current_y = y_median
            while True:
                if get_h(current_y) <= d_prime:
                    y_upper_bound_y = current_y
                    current_y += 1
                else:
                    break
            if y_upper_bound_y is None:
                y_upper_bound_y = y_median
                
            if y_lower_bound_y is not None and y_upper_bound_y is not None:
                count += max(0, y_upper_bound_y - y_lower_bound_y + 1)
                
    print(count)

if __name__ == '__main__':
    solve()