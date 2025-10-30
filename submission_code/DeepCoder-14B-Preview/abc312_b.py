n, m = map(int, input().split())
grid = [input().strip() for _ in range(n)]

# Create the 9x9 mask
mask = [['X' for _ in range(9)] for _ in range(9)]

# Mark top-left 3x3 as 'B'
for i in range(3):
    for j in range(3):
        mask[i][j] = 'B'

# Mark bottom-right 3x3 as 'B'
for i in range(6, 9):
    for j in range(6, 9):
        mask[i][j] = 'B'

# Mark top-left adjacent region as 'W'
for i in range(4):
    for j in range(4):
        if i < 3 and j < 3:
            continue
        mask[i][j] = 'W'

# Mark bottom-right adjacent region as 'W'
for i in range(5, 9):
    for j in range(5, 9):
        if i >= 6 and j >= 6:
            continue
        mask[i][j] = 'W'

valid_regions = []

# Iterate over all possible top-left cells of 9x9 regions
for i in range(n - 8):
    for j in range(m - 8):
        valid = True
        for x in range(9):
            if not valid:
                break
            for y in range(9):
                cell = grid[i + x][j + y]
                m_val = mask[x][y]
                if m_val == 'B' and cell != '#':
                    valid = False
                    break
                if m_val == 'W' and cell != '.':
                    valid = False
                    break
        if valid:
            valid_regions.append((i + 1, j + 1))  # Convert to 1-based index

# Sort the valid regions lexicographically
valid_regions.sort()

# Print the results
for i, j in valid_regions:
    print(i, j)