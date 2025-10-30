# Read the input
H, W = map(int, input().split())
grid = [input() for _ in range(H)]

# Check if it's possible to form a rectangle
has_black = False
for i in range(H):
    for j in range(W):
        if grid[i][j] == '#':
            has_black = True
            break
    if has_black:
        break

if not has_black:
    print("No")
    exit()

# Find the top-left and bottom-right corners of the rectangle
top, left = None, None
bottom, right = None, None
for i in range(H):
    for j in range(W):
        if grid[i][j] == '#':
            if top is None or i < top:
                top = i
            if left is None or j < left:
                left = j
            if bottom is None or i > bottom:
                bottom = i
            if right is None or j > right:
                right = j

# Check if all the remaining cells can be painted to form a rectangle
for i in range(H):
    for j in range(W):
        if (i < top or i > bottom or j < left or j > right) and grid[i][j] == '?':
            grid[i] = grid[i][:j] + '.' + grid[i][j+1:]
        elif (top <= i <= bottom and left <= j <= right) and grid[i][j] == '?':
            grid[i] = grid[i][:j] + '#' + grid[i][j+1:]

# Print the result
print("Yes")