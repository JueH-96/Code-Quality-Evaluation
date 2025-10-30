# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])
X = int(data[2]) - 1
Y = int(data[3]) - 1

grid = []
for i in range(H):
    grid.append(list(data[4 + i]))

T = data[4 + H]

visited_houses = set()

def is_passable(x, y):
    return grid[x][y] != '#'

def move(x, y, direction):
    if direction == 'U' and is_passable(x - 1, y):
        return x - 1, y
    elif direction == 'D' and is_passable(x + 1, y):
        return x + 1, y
    elif direction == 'L' and is_passable(x, y - 1):
        return x, y - 1
    elif direction == 'R' and is_passable(x, y + 1):
        return x, y + 1
    return x, y

current_x, current_y = X, Y

for direction in T:
    current_x, current_y = move(current_x, current_y, direction)
    if grid[current_x][current_y] == '@':
        visited_houses.add((current_x, current_y))

print(current_x + 1, current_y + 1, len(visited_houses))