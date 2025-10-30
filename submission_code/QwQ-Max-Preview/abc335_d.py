n = int(input())
grid = [[0] * n for _ in range(n)]
c = (n - 1) // 2
center = (c, c)

result = []
top = 0
bottom = n - 1
left = 0
right = n - 1

while top <= bottom and left <= right:
    # Process top row
    for i in range(left, right + 1):
        if (top, i) != center:
            result.append((top, i))
    top += 1

    # Process right column
    for i in range(top, bottom + 1):
        if (i, right) != center:
            result.append((i, right))
    right -= 1

    if top > bottom or left > right:
        break

    # Process bottom row
    for i in range(right, left - 1, -1):
        if (bottom, i) != center:
            result.append((bottom, i))
    bottom -= 1

    # Process left column
    for i in range(bottom, top - 1, -1):
        if (i, left) != center:
            result.append((i, left))
    left += 1

# Fill the grid with numbers
for num, (i, j) in enumerate(result, start=1):
    grid[i][j] = num

# Place 'T' in the center
grid[center[0]][center[1]] = 'T'

# Print the grid
for row in grid:
    print(' '.join(map(str, row)))