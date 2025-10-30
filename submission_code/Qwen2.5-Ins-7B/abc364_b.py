# YOUR CODE HERE
h, w = map(int, input().split())
si, sj = map(int, input().split())
grid = [input() for _ in range(h)]
x, y = sj-1, si-1
for move in input():
    if move == 'L' and y > 0 and grid[x][y-1] == '.':
        y -= 1
    elif move == 'R' and y < w-1 and grid[x][y+1] == '.':
        y += 1
    elif move == 'U' and x > 0 and grid[x-1][y] == '.':
        x -= 1
    elif move == 'D' and x < h-1 and grid[x+1][y] == '.':
        x += 1
print(x+1, y+1)