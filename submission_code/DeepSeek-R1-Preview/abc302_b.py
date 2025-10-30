H, W = map(int, input().split())
grid = [input().strip() for _ in range(H)]

# Define all possible directions (dx, dy)
directions = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),          (0, 1),
              (1, -1),  (1, 0), (1, 1)]

for i in range(H):
    for j in range(W):
        if grid[i][j] != 's':
            continue
        # Check each direction
        for dx, dy in directions:
            valid = True
            positions = []
            # Starting position (i+1, j+1)
            positions.append((i + 1, j + 1))
            for step in range(1, 5):
                ni = i + dx * step
                nj = j + dy * step
                if ni < 0 or ni >= H or nj < 0 or nj >= W:
                    valid = False
                    break
                current_char = grid[ni][nj]
                expected_char = ['n', 'u', 'k', 'e'][step - 1]
                if current_char != expected_char:
                    valid = False
                    break
                positions.append((ni + 1, nj + 1))
            if valid:
                for pos in positions:
                    print(pos[0], pos[1])
                exit()