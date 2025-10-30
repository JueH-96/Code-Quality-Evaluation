def find_valid_coords(points, D):
    points_sorted = sorted(points)
    n = len(points_sorted)
    
    # Precompute prefix sums
    prefix_sum = [0] * (n + 1)
    for i in range(n):
        prefix_sum[i + 1] = prefix_sum[i] + points_sorted[i]
    
    def compute_distance_sum(x):
        # Binary search to find the number of points < x
        left, right = 0, n
        while left < right:
            mid = (left + right) // 2
            if points_sorted[mid] < x:
                left = mid + 1
            else:
                right = mid
        
        k = left  # Number of points < x
        
        # Compute the sum
        sum_left = k * x - prefix_sum[k]
        sum_right = (prefix_sum[n] - prefix_sum[k]) - (n - k) * x
        
        return sum_left + sum_right
    
    # Find the median
    median = points_sorted[n // 2]
    
    valid_coords = {}
    
    # Start from the median and expand
    coord = median
    dist = compute_distance_sum(coord)
    if dist <= D:
        valid_coords[coord] = dist
    
    # Expand to the left
    coord = median - 1
    while True:
        dist = compute_distance_sum(coord)
        if dist > D:
            break
        valid_coords[coord] = dist
        coord -= 1
    
    # Expand to the right
    coord = median + 1
    while True:
        dist = compute_distance_sum(coord)
        if dist > D:
            break
        valid_coords[coord] = dist
        coord += 1
    
    return valid_coords

N, D = map(int, input().split())
x_coords = []
y_coords = []
for _ in range(N):
    x, y = map(int, input().split())
    x_coords.append(x)
    y_coords.append(y)

x_distances = find_valid_coords(x_coords, D)
y_distances = find_valid_coords(y_coords, D)

count = 0
for x, dx in x_distances.items():
    for y, dy in y_distances.items():
        if dx + dy <= D:
            count += 1

print(count)