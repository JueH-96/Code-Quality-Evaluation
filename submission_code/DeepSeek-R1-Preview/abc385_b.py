h, w, x, y = map(int, input().split())
grid = [input().strip() for _ in range(h)]
t = input().strip()

curr_x, curr_y = x, y
houses = set()

for move in t:
    next_x, next_y = curr_x, curr_y
    if move == 'U':
        next_x -= 1
    elif move == 'D':
        next_x += 1
    elif move == 'L':
        next_y -= 1
    elif move == 'R':
        next_y += 1

    # Check if the next position is within bounds and passable
    if 1 <= next_x <= h and 1 <= next_y <= w:
        if grid[next_x - 1][next_y - 1] != '#':
            curr_x, curr_y = next_x, next_y

    # Check current cell for house
    if grid[curr_x - 1][curr_y - 1] == '@':
        houses.add((curr_x, curr_y))

print(curr_x, curr_y, len(houses))