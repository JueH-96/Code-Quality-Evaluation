def get_possible_placements(grid, tile, H, W):
    placements = []
    # Orientation 0: A_i x B_i
    for r in range(H - tile[0] + 1):
        for c in range(W - tile[1] + 1):
            if all(grid[r + dr][c + dc] == 0 for dr in range(tile[0]) for dc in range(tile[1])):
                placements.append((0, (r, c)))
    # Orientation 1: B_i x A_i
    if tile[0] != tile[1]:
        for r in range(H - tile[1] + 1):
            for c in range(W - tile[0] + 1):
                if all(grid[r + dr][c + dc] == 0 for dr in range(tile[1]) for dc in range(tile[0])):
                    placements.append((1, (r, c)))
    return placements

def place_tile(grid, tile, orientation, top_left, H, W):
    row, col = top_left
    if orientation == 0:
        height, width = tile
    else:
        width, height = tile
    if row + height > H or col + width > W:
        return False
    for r in range(height):
        for c in range(width):
            if grid[row + r][col + c] != 0:
                return False
    for r in range(height):
        for c in range(width):
            grid[row + r][col + c] = 1
    return True

def remove_tile(grid, tile, orientation, top_left, H, W):
    row, col = top_left
    if orientation == 0:
        height, width = tile
    else:
        width, height = tile
    for r in range(height):
        for c in range(width):
            grid[row + r][col + c] = 0

def is_grid_covered(grid):
    for row in grid:
        for cell in row:
            if cell == 0:
                return False
    return True

def can_place_subset(tiles, grid, placements):
    if not tiles:
        return is_grid_covered(grid)
    else:
        # Choose the tile with the fewest placements
        tile = min(tiles, key=lambda t: len(placements[t]))
        for orientation, position in placements[tile]:
            if place_tile(grid, tile, orientation, position, H, W):
                if can_place_subset(tiles - {tile}, grid, placements):
                    return True
                remove_tile(grid, tile, orientation, position, H, W)
        return False

def main():
    import sys
    N, H, W = map(int, sys.stdin.readline().split())
    tiles = []
    for _ in range(N):
        A, B = map(int, sys.stdin.readline().split())
        tiles.append((A, B))
    for mask in range(1 << N):
        subset = [tiles[i] for i in range(N) if (mask >> i) & 1]
        total_area = sum(a * b for a, b in subset)
        if total_area == H * W:
            grid = [[0]*W for _ in range(H)]
            placements = {tile: get_possible_placements(grid, tile, H, W) for tile in subset}
            if can_place_subset(set(subset), grid, placements):
                print("Yes")
                return
    print("No")

if __name__ == '__main__':
    main()