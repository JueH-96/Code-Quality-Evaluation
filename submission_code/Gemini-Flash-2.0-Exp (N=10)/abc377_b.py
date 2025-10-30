grid = []
for _ in range(8):
    grid.append(input())

count = 0
for r in range(8):
    for c in range(8):
        if grid[r][c] == '.':
            safe = True
            for i in range(8):
                if grid[r][i] == '#':
                    safe = False
                    break
            if safe:
                for i in range(8):
                    if grid[i][c] == '#':
                        safe = False
                        break
            if safe:
                count += 1
print(count)