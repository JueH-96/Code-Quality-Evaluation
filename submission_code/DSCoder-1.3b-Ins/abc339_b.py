H, W, N = map(int, input().split())

grid = [['.' for _ in range(W+1)] for _ in range(H+1)]

directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
current_direction = 0
current_pos = (1, 1)

for _ in range(N):
    if grid[current_pos[0]][current_pos[1]] == '.':
        grid[current_pos[0]][current_pos[1]] = '#'
    else:
        grid[current_pos[0]][current_pos[1]] = '.'

    new_direction = (current_direction + 1) % 4
    current_pos = (current_pos[0] + directions[new_direction][0], current_pos[1] + directions[new_direction][1])
    current_direction = new_direction

for row in grid[1:]:
    print(''.join(row))