def check_house_on_segment(house_x, house_y, x1, y1, x2, y2):
    """Check if a house at (house_x, house_y) is on the line segment from (x1, y1) to (x2, y2)"""
    # For horizontal movement (y1 == y2)
    if y1 == y2:
        if house_y == y1:
            min_x = min(x1, x2)
            max_x = max(x1, x2)
            return min_x <= house_x <= max_x
        return False
    
    # For vertical movement (x1 == x2)
    if x1 == x2:
        if house_x == x1:
            min_y = min(y1, y2)
            max_y = max(y1, y2)
            return min_y <= house_y <= max_y
        return False
    
    return False

# Read input
N, M, S_x, S_y = map(int, input().split())

houses = []
for _ in range(N):
    x, y = map(int, input().split())
    houses.append((x, y))

# Current position
curr_x, curr_y = S_x, S_y

# Set to store visited houses
visited_houses = set()

# Process each movement
for _ in range(M):
    parts = input().split()
    direction = parts[0]
    distance = int(parts[1])
    
    # Calculate new position
    new_x, new_y = curr_x, curr_y
    
    if direction == 'U':
        new_y = curr_y + distance
    elif direction == 'D':
        new_y = curr_y - distance
    elif direction == 'L':
        new_x = curr_x - distance
    elif direction == 'R':
        new_x = curr_x + distance
    
    # Check which houses are on this line segment
    for i, (house_x, house_y) in enumerate(houses):
        if check_house_on_segment(house_x, house_y, curr_x, curr_y, new_x, new_y):
            visited_houses.add(i)
    
    # Update current position
    curr_x, curr_y = new_x, new_y

# Output results
print(curr_x, curr_y, len(visited_houses))