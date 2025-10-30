# YOUR CODE HERE
n = int(input())
grid = [['' for _ in range(n)] for _ in range(n)]
center = (n // 2, n // 2)
grid[center[0]][center[1]] = 'T'
current = 1
direction = 0  # 0: right, 1: down, 2: left, 3: up
row, col = center[0], center[1] - 1
for layer in range(1, n // 2 + 1):
    for _ in range(2):
        for _ in range(layer):
            if current > n * n - 1:
                break
            grid[row][col] = str(current)
            current += 1
            if direction == 0:
                col += 1
            elif direction == 1:
                row += 1
            elif direction == 2:
                col -= 1
            elif direction == 3:
                row -= 1
        direction = (direction + 1) % 4
    if direction == 0:
        row += 1
        col -=1
    elif direction == 1:
        row -=1
        col -=1
    elif direction == 2:
        row -=1
        col +=1
    elif direction == 3:
        row +=1
        col +=1
for row in grid:
    print(' '.join(row))