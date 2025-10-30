# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])
grid = data[2:]

# Find the top-left corner of the rectangle
top_left = None
for i in range(H):
    for j in range(W):
        if grid[i][j] == '#':
            top_left = (i, j)
            break
    if top_left:
        break

# Find the bottom-right corner of the rectangle
bottom_right = None
for i in range(H-1, -1, -1):
    for j in range(W-1, -1, -1):
        if grid[i][j] == '#':
            bottom_right = (i, j)
            break
    if bottom_right:
        break

# Find the missing cookie
missing_cookie = None
for i in range(top_left[0], bottom_right[0] + 1):
    for j in range(top_left[1], bottom_right[1] + 1):
        if grid[i][j] == '.':
            missing_cookie = (i + 1, j + 1)
            break
    if missing_cookie:
        break

print(missing_cookie[0], missing_cookie[1])