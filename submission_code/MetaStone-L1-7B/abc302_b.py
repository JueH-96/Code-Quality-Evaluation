H, W = map(int, input().split())
grid = [input().strip() for _ in range(H)]
directions = [ (0, 1), (1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1), (0, -1), (-1, 0) ]
target = ['n', 'u', 'k', 'e']

for r in range(H):
    for c in range(W):
        if grid[r][c] == 's':
            for dr, dc in directions:
                valid = True
                for i in range(1, 5):
                    nr = r + dr * i
                    nc = c + dc * i
                    if nr < 0 or nr >= H or nc < 0 or nc >= W:
                        valid = False
                        break
                    if grid[nr][nc] != target[i-1]:
                        valid = False
                        break
                if valid:
                    print(r + 1, c + 1)
                    print(r + dr * 1 + 1, c + dc * 1 + 1)
                    print(r + dr * 2 + 1, c + dc * 2 + 1)
                    print(r + dr * 3 + 1, c + dc * 3 + 1)
                    print(r + dr * 4 + 1, c + dc * 4 + 1)
                    exit()