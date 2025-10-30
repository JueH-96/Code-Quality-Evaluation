# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])
grid = data[2:]

# Find the top-left corner of the rectangle
for i in range(H):
    for j in range(W):
        if grid[i][j] == '#':
            a = i
            c = j
            break
    else:
        continue
    break

# Find the bottom-right corner of the rectangle
for i in range(H-1, -1, -1):
    for j in range(W-1, -1, -1):
        if grid[i][j] == '#':
            b = i
            d = j
            break
    else:
        continue
    break

# Find the square that was eaten
for i in range(a, b+1):
    for j in range(c, d+1):
        if grid[i][j] == '.':
            print(i+1, j+1)
            exit()