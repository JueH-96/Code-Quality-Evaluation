# H W Y = map(int, input().split())
# A = [list(map(int, input().split())) for _ in range(H)]
H, W, Y = map(int, input().split())
A = [list(map(int, input().split())) for _ in range(H)]

def flood_fill(i, j, sea_level):
    if i < 0 or i >= H or j < 0 or j >= W or A[i][j] <= sea_level or visited[i][j]:
        return
    visited[i][j] = True
    flood_fill(i-1, j, sea_level)
    flood_fill(i+1, j, sea_level)
    flood_fill(i, j-1, sea_level)
    flood_fill(i, j+1, sea_level)

def simulate(sea_level):
    global remaining_area
    visited = [[False] * W for _ in range(H)]
    for i in range(H):
        for j in range(W):
            if A[i][j] > sea_level and not visited[i][j]:
                flood_fill(i, j, sea_level)
                remaining_area += 1

remaining_area = 0
for y in range(Y + 1):
    simulate(y)
    print(remaining_area)