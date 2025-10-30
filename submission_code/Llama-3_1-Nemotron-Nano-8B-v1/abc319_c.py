import itertools
from collections import Counter

# Read input
grid = []
for _ in range(3):
    row = list(map(int, input().split()))
    grid.extend(row)

# Generate all unique permutations of the grid's values
unique_perms = set(itertools.permutations(grid))

# Generate all lines
lines = []

# Rows
for i in range(3):
    lines.append(grid[i*3 : (i+1)*3])

# Columns
for j in range(3):
    lines.append([grid[i*3 + j] for i in range(3)])

# Diagonals
lines.append([grid[i] for i in range(3) if i % 2 == 0])  # (0,0), (1,1), (2,2)
lines.append([grid[i*3-2] for i in range(3)])  # (0,2), (1,1), (2,0)

# Convert lines to tuples for hashing in set
lines = [tuple(line) for line in lines]

valid_count = 0

for perm in unique_perms:
    valid = True
    for line in lines:
        cnt = Counter(line)
        if len(cnt) != 2:
            continue
        x, y = cnt.keys()
        if cnt[x] == 1:
            x, y = y, x
        # Find positions of x and y in the current permutation
        x_pos = [i for i, num in enumerate(perm) if num == x]
        y_pos = [i for i, num in enumerate(perm) if num == y][0]
        x1, x2 = sorted(x_pos)
        if x2 < y_pos:
            valid = False
            break
    if valid:
        valid_count += 1

total_perms = len(unique_perms)
probability = valid_count / total_perms
print("{0:.12f}".format(probability))