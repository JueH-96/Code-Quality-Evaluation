def can_place_tile(grid, H, W, tile, x, y):
    A, B = tile
    if x + A > H or y + B > W:
        return False
    for i in range(A):
        for j in range(B):
            if grid[x + i][y + j] != 0:
                return False
    return True

def place_tile(grid, tile, x, y, value):
    A, B = tile
    for i in range(A):
        for j in range(B):
            grid[x + i][y + j] = value

def backtrack(grid, H, W, tiles, index):
    if index == len(tiles):
        return all(all(cell != 0 for cell in row) for row in grid)
    
    for x in range(H):
        for y in range(W):
            for orientation in [(tiles[index][0], tiles[index][1]), (tiles[index][1], tiles[index][0])]:
                if can_place_tile(grid, H, W, orientation, x, y):
                    place_tile(grid, orientation, x, y, index + 1)
                    if backtrack(grid, H, W, tiles, index + 1):
                        return True
                    place_tile(grid, orientation, x, y, 0)
    
    return False

def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    H = int(data[1])
    W = int(data[2])
    
    tiles = []
    for i in range(N):
        A = int(data[3 + 2 * i])
        B = int(data[3 + 2 * i + 1])
        tiles.append((A, B))
    
    grid = [[0] * W for _ in range(H)]
    
    if backtrack(grid, H, W, tiles, 0):
        print("Yes")
    else:
        print("No")