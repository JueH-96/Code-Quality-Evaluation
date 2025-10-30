H, W = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input().strip())

target = "snuke"
directions = [
    (0, 1),   # right
    (0, -1),  # left
    (1, 0),   # down
    (-1, 0),  # up
    (1, 1),   # down-right
    (1, -1),  # down-left
    (-1, 1),  # up-right
    (-1, -1)  # up-left
]

def is_valid(r, c):
    return 0 <= r < H and 0 <= c < W

def search():
    for i in range(H):
        for j in range(W):
            for di, dj in directions:
                positions = []
                valid = True
                for k in range(5):
                    ni, nj = i + k * di, j + k * dj
                    if not is_valid(ni, nj):
                        valid = False
                        break
                    if grid[ni][nj] != target[k]:
                        valid = False
                        break
                    positions.append((ni + 1, nj + 1))  # 1-indexed
                
                if valid:
                    return positions
    return None

result = search()
for r, c in result:
    print(r, c)