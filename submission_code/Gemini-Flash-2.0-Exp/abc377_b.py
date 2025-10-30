grid = []
for _ in range(8):
    grid.append(input())

count = 0
for i in range(8):
    for j in range(8):
        if grid[i][j] == '.':
            can_place = True
            # Check row
            for k in range(8):
                if grid[i][k] == '#':
                    can_place = False
                    break
            if not can_place:
                continue
            # Check column
            for k in range(8):
                if grid[k][j] == '#':
                    can_place = False
                    break
            if can_place:
                count += 1

print(count)