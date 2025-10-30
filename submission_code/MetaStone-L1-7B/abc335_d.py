n = int(input())
c = (n + 1) // 2
grid = [['T' for _ in range(n + 1)] for _ in range(n + 1)]
directions = [(0, 1), (-1, 0), (0, -1), (1, 0)]
current_dir = 0
step = 1
current_pos = (c, c)
filled = set()
filled.add(current_pos)
grid[current_pos[0]][current_pos[1]] = 1

for x in range(2, n * n + 1):
    for _ in range(step):
        next_pos = (current_pos[0] + directions[current_dir][0], current_pos[1] + directions[current_dir][1])
        if 1 <= next_pos[0] <= n and 1 <= next_pos[1] <= n and next_pos not in filled:
            current_pos = next_pos
            filled.add(next_pos)
            grid[current_pos[0]][current_pos[1]] = x
        else:
            break
    current_dir = (current_dir + 1) % 4
    if (x - 2) % 2 == 0:
        step += 1

for row in grid:
    print(' '.join(row))