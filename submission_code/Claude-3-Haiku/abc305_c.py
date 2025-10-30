# Read input
H, W = map(int, input().split())
grid = [list(input()) for _ in range(H)]

# Find the missing cookie
for i in range(H):
    for j in range(W):
        if grid[i][j] == '.':
            # Check if the current square is inside the rectangle
            for a in range(1, H):
                for b in range(a+1, H+1):
                    for c in range(1, W):
                        for d in range(c+1, W+1):
                            if all(grid[x-1][y-1] == '#' for x in range(a, b+1) for y in range(c, d+1)):
                                print(i+1, j+1)
                                return