def solve():
    N, D = map(int, input().split())
    points = []
    for _ in range(N):
        x, y = map(int, input().split())
        points.append((x, y))
    
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    # Sort coordinates to find median
    x_sorted = sorted(x_coords)
    y_sorted = sorted(y_coords)
    
    # Find the median (which minimizes the sum of Manhattan distances)
    if N % 2 == 0:
        x_med = (x_sorted[N//2 - 1] + x_sorted[N//2]) // 2
        y_med = (y_sorted[N//2 - 1] + y_sorted[N//2]) // 2
    else:
        x_med = x_sorted[N//2]
        y_med = y_sorted[N//2]
    
    # Function to compute sum of Manhattan distances for a given x coordinate
    def g_x(x):
        return sum(abs(x - x_i) for x_i in x_coords)
    
    # Function to compute sum of Manhattan distances for a given y coordinate
    def g_y(y):
        return sum(abs(y - y_i) for y_i in y_coords)
    
    # Minimum values of g_x and g_y
    g_x_min = g_x(x_med)
    g_y_min = g_y(y_med)
    
    # If the minimum sum of Manhattan distances exceeds D, no valid pairs exist
    if g_x_min + g_y_min > D:
        return 0
    
    # Binary search to find the leftmost valid x
    def binary_search_left(func, median, threshold, range_size=10**6):
        left, right = median - range_size, median
        while left < right:
            mid = (left + right) // 2
            if func(mid) <= threshold:
                right = mid
            else:
                left = mid + 1
        return left
    
    # Binary search to find the rightmost valid x
    def binary_search_right(func, median, threshold, range_size=10**6):
        left, right = median, median + range_size
        while left < right:
            mid = (left + right + 1) // 2
            if func(mid) <= threshold:
                left = mid
            else:
                right = mid - 1
        return right
    
    # Find the range of valid x values
    x_min = binary_search_left(g_x, x_med, D)
    x_max = binary_search_right(g_x, x_med, D)
    
    # Count valid pairs
    count = 0
    for x in range(x_min, x_max + 1):
        g_x_val = g_x(x)
        remaining = D - g_x_val
        
        # Find the range of valid y values for this x
        y_min = binary_search_left(g_y, y_med, remaining)
        y_max = binary_search_right(g_y, y_med, remaining)
        
        count += (y_max - y_min + 1)
    
    return count

print(solve())