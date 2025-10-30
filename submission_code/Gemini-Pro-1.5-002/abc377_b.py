grid = []
for _ in range(8):
    grid.append(input())

count = 0
for i in range(8):
    for j in range(8):
        if grid[i][j] == '.':
            safe = True
            for k in range(8):
                if grid[i][k] == '#':
                    safe = False
                    break
                if grid[k][j] == '#':
                    safe = False
                    break
            if safe:
                count += 1

print(count)