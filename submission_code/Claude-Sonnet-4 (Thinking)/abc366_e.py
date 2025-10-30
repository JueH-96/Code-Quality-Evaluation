n, d = map(int, input().split())
points = []
for _ in range(n):
    x, y = map(int, input().split())
    points.append((x, y))

x_coords = [p[0] for p in points]
y_coords = [p[1] for p in points]

x_min, x_max = min(x_coords), max(x_coords)
y_min, y_max = min(y_coords), max(y_coords)

x_coords.sort()
y_coords.sort()

def compute_f_values(coords, min_coord, max_coord, d):
    f_values = []
    start = min_coord - d
    end = max_coord + d
    
    # Compute f(start)
    f_val = sum(abs(start - ci) for ci in coords)
    f_values.append(f_val)
    
    count_less = 0
    coord_index = 0
    
    for x in range(start + 1, end + 1):
        # Update count_less
        while coord_index < len(coords) and coords[coord_index] < x:
            count_less += 1
            coord_index += 1
        
        # f(x) = f(x-1) + 2 * count_less - n
        f_val = f_val + 2 * count_less - len(coords)
        f_values.append(f_val)
    
    return f_values

f_x_values = compute_f_values(x_coords, x_min, x_max, d)
f_y_values = compute_f_values(y_coords, y_min, y_max, d)

f_y_values.sort()

count = 0
for f_x_val in f_x_values:
    target = d - f_x_val
    if target >= 0:
        # Binary search for the number of f_y_val <= target
        left, right = 0, len(f_y_values)
        while left < right:
            mid = (left + right) // 2
            if f_y_values[mid] <= target:
                left = mid + 1
            else:
                right = mid
        count += left

print(count)