def is_between(a, b, c):
    # Returns True if c is between a and b (inclusive)
    return min(a, b) <= c <= max(a, b)

def check_line_intersection(x1, y1, x2, y2, house_x, house_y):
    # Check if house point lies on the line segment between (x1,y1) and (x2,y2)
    if x1 == x2:  # Vertical line
        return house_x == x1 and is_between(y1, y2, house_y)
    elif y1 == y2:  # Horizontal line
        return house_y == y1 and is_between(x1, x2, house_x)
    return False

# Read input
N, M, Sx, Sy = map(int, input().split())
houses = []
for _ in range(N):
    x, y = map(int, input().split())
    houses.append((x, y))

# Process moves
curr_x, curr_y = Sx, Sy
visited_houses = set()

for _ in range(M):
    D, C = input().split()
    C = int(C)
    
    # Calculate next position
    next_x, next_y = curr_x, curr_y
    if D == 'U':
        next_y = curr_y + C
    elif D == 'D':
        next_y = curr_y - C
    elif D == 'L':
        next_x = curr_x - C
    else:  # R
        next_x = curr_x + C
    
    # Check houses passed through in this move
    for house_x, house_y in houses:
        if check_line_intersection(curr_x, curr_y, next_x, next_y, house_x, house_y):
            visited_houses.add((house_x, house_y))
    
    # Update current position
    curr_x, curr_y = next_x, next_y

# Print result
print(curr_x, curr_y, len(visited_houses))