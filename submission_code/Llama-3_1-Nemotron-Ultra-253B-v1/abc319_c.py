import itertools

grid = []
for _ in range(3):
    row = list(map(int, input().split()))
    grid.append(row)

cells = [(i, j) for i in range(3) for j in range(3)]

all_lines = [
    [(0,0), (0,1), (0,2)],
    [(1,0), (1,1), (1,2)],
    [(2,0), (2,1), (2,2)],
    [(0,0), (1,0), (2,0)],
    [(0,1), (1,1), (2,1)],
    [(0,2), (1,2), (2,2)],
    [(0,0), (1,1), (2,2)],
    [(2,0), (1,1), (0,2)]
]

active_lines = []
for line in all_lines:
    values = [grid[i][j] for (i,j) in line]
    if len(set(values)) == 2:
        active_lines.append(line)

count = 0
for perm in itertools.permutations(cells):
    position_dict = {cell: idx for idx, cell in enumerate(perm)}
    valid = True
    for line in active_lines:
        indices = [position_dict[cell] for cell in line]
        sorted_indices = sorted(indices)
        ordered_values = [grid[perm[i][0]][perm[i][1]] for i in sorted_indices]
        if ordered_values[0] == ordered_values[1] and ordered_values[1] != ordered_values[2]:
            valid = False
            break
    if valid:
        count += 1

total = 362880  # 9! = 362880
print("{0:.15f}".format(count / total))