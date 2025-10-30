import sys
data = sys.stdin.read().split()
index = 0
H = int(data[index])
index += 1
W = int(data[index])
index += 1
start_row = int(data[index])
index += 1
start_col = int(data[index])
index += 1
grid = []
for i in range(H):
    row_str = data[index]
    grid.append(row_str)
    index += 1
T_str = data[index]
# Initialize current position and house set
curr_x = start_row
curr_y = start_col
house_set = set()
# Process each move in T_str
for char in T_str:
    if char == 'U':
        dx, dy = -1, 0
    elif char == 'D':
        dx, dy = 1, 0
    elif char == 'L':
        dx, dy = 0, -1
    elif char == 'R':
        dx, dy = 0, 1
    new_x = curr_x + dx
    new_y = curr_y + dy
    # Check if new position is within bounds and passable
    if 1 <= new_x <= H and 1 <= new_y <= W and grid[new_x - 1][new_y - 1] != '#':
        curr_x = new_x
        curr_y = new_y
    # After move or stay, check if current cell has a house
    if grid[curr_x - 1][curr_y - 1] == '@':
        house_set.add((curr_x, curr_y))
# Calculate the number of distinct houses
C = len(house_set)
# Output the final position and count
print(curr_x, curr_y, C)