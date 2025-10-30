grid = [input() for _ in range(8)]
count = 0
for i in range(8):
    for j in range(8):
        if grid[i][j] == '.':
            safe = True
            for k in range(8):
                if grid[i][k] == '#' and k != j:
                    safe = False
                    break
            if safe:
                for k in range(8):
                    if grid[k][j] == '#' and k != i:
                        safe = False
                        break
            if safe:
                count += 1
print(count)