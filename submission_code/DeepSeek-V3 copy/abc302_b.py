# YOUR CODE HERE
H, W = map(int, input().split())
grid = [input().strip() for _ in range(H)]
target = ['s', 'n', 'u', 'k', 'e']

# Directions: 8 possible directions (horizontal, vertical, diagonal)
directions = [ (1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1) ]

for i in range(H):
    for j in range(W):
        if grid[i][j] == 's':
            for dx, dy in directions:
                x, y = i, j
                sequence = []
                for k in range(5):
                    if x < 0 or x >= H or y < 0 or y >= W:
                        break
                    if grid[x][y] != target[k]:
                        break
                    sequence.append((x+1, y+1))  # Convert to 1-based indexing
                    x += dx
                    y += dy
                if len(sequence) == 5:
                    for pos in sequence:
                        print(pos[0], pos[1])
                    exit()