def count_valid_y(y_coords, budget):
    """Count integer y values with sum of |y - y_i| <= budget"""
    if budget < 0:
        return 0
    
    n = len(y_coords)
    sorted_y = sorted(y_coords)
    
    # Find the median range
    if n % 2 == 1:
        median = sorted_y[n // 2]
        min_sum = sum(abs(y - median) for y in y_coords)
    else:
        # Any value between the two middle elements gives the same minimum sum
        med1 = sorted_y[n // 2 - 1]
        med2 = sorted_y[n // 2]
        min_sum = sum(abs(y - med1) for y in y_coords)
    
    if min_sum > budget:
        return 0
    
    # Binary search for the range of valid y values
    # The sum increases as we move away from the median
    
    # Find minimum valid y
    left, right = -10**7, sorted_y[0]
    while left < right:
        mid = (left + right) // 2
        s = sum(abs(mid - y) for y in y_coords)
        if s <= budget:
            right = mid
        else:
            left = mid + 1
    min_y = left
    
    # Find maximum valid y
    left, right = sorted_y[-1], 10**7
    while left < right:
        mid = (left + right + 1) // 2
        s = sum(abs(mid - y) for y in y_coords)
        if s <= budget:
            left = mid
        else:
            right = mid - 1
    max_y = left
    
    return max_y - min_y + 1

# Read input
N, D = map(int, input().split())
points = []
for _ in range(N):
    x, y = map(int, input().split())
    points.append((x, y))

x_coords = [p[0] for p in points]
y_coords = [p[1] for p in points]

# Determine x range to check
min_x = min(x_coords)
max_x = max(x_coords)

# Extend the range based on D
x_range = D // N + 1
check_min_x = min_x - x_range
check_max_x = max_x + x_range

result = 0

# For each x, calculate sum of |x - x_i| and find valid y values
for x in range(check_min_x, check_max_x + 1):
    sum_x = sum(abs(x - xi) for xi in x_coords)
    if sum_x > D:
        continue
    
    remaining_budget = D - sum_x
    result += count_valid_y(y_coords, remaining_budget)

print(result)