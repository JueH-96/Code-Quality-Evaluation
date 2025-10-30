# YOUR CODE HERE
N, M, Sx, Sy = map(int, input().split())

houses = []
for _ in range(N):
    x, y = map(int, input().split())
    houses.append((x, y))

moves = []
for _ in range(M):
    line = input().split()
    d = line[0]
    c = int(line[1])
    moves.append((d, c))

# Track visited houses
visited_houses = set()

# Current position
curr_x, curr_y = Sx, Sy

# Process each move
for direction, distance in moves:
    start_x, start_y = curr_x, curr_y
    
    # Calculate end position
    if direction == 'U':
        end_x, end_y = curr_x, curr_y + distance
    elif direction == 'D':
        end_x, end_y = curr_x, curr_y - distance
    elif direction == 'L':
        end_x, end_y = curr_x - distance, curr_y
    elif direction == 'R':
        end_x, end_y = curr_x + distance, curr_y
    
    # Check which houses lie on this path
    for hx, hy in houses:
        if direction in ['L', 'R']:  # Horizontal movement
            if hy == start_y:  # Same y-coordinate
                if min(start_x, end_x) <= hx <= max(start_x, end_x):
                    visited_houses.add((hx, hy))
        else:  # Vertical movement (U or D)
            if hx == start_x:  # Same x-coordinate
                if min(start_y, end_y) <= hy <= max(start_y, end_y):
                    visited_houses.add((hx, hy))
    
    # Update current position
    curr_x, curr_y = end_x, end_y

print(curr_x, curr_y, len(visited_houses))