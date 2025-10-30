# Read input
N, M, S_x, S_y = map(int, input().split())

houses = []
for _ in range(N):
    x, y = map(int, input().split())
    houses.append((x, y))

movements = []
for _ in range(M):
    parts = input().split()
    direction = parts[0]
    count = int(parts[1])
    movements.append((direction, count))

# Track visited houses
visited_houses = set()

# Current position
current_x, current_y = S_x, S_y

# Process each movement
for direction, distance in movements:
    start_x, start_y = current_x, current_y
    
    # Calculate end position
    if direction == 'U':
        end_x, end_y = current_x, current_y + distance
    elif direction == 'D':
        end_x, end_y = current_x, current_y - distance
    elif direction == 'L':
        end_x, end_y = current_x - distance, current_y
    elif direction == 'R':
        end_x, end_y = current_x + distance, current_y
    
    # Check which houses are on the path
    for house_x, house_y in houses:
        if direction in ['L', 'R']:  # Horizontal movement
            if house_y == start_y:  # Same y-coordinate
                if min(start_x, end_x) <= house_x <= max(start_x, end_x):
                    visited_houses.add((house_x, house_y))
        else:  # Vertical movement (U or D)
            if house_x == start_x:  # Same x-coordinate
                if min(start_y, end_y) <= house_y <= max(start_y, end_y):
                    visited_houses.add((house_x, house_y))
    
    # Update current position
    current_x, current_y = end_x, end_y

# Output result
print(current_x, current_y, len(visited_houses))