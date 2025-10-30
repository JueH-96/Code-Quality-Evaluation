def find_snake():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    grid = [data[i + 2] for i in range(H)]
    
    directions = [
        (0, 1),   # right
        (1, 0),   # down
        (1, 1),   # down-right diagonal
        (-1, 1),  # up-right diagonal
    ]
    
    target = "snuke"
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 's':
                for dx, dy in directions:
                    # Check if we can fit the word "snuke" in this direction
                    valid = True
                    positions = []
                    for k in range(5):
                        ni = i + k * dx
                        nj = j + k * dy
                        if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == target[k]:
                            positions.append((ni + 1, nj + 1))  # Convert to 1-based index
                        else:
                            valid = False
                            break
                    if valid:
                        for pos in positions:
                            print(pos[0], pos[1])
                        return