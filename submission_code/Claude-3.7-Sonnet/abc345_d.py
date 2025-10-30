def can_place_tiles(grid, available_tiles, h, w):
    # Find the next empty cell
    row, col = -1, -1
    for i in range(h):
        for j in range(w):
            if not grid[i][j]:
                row, col = i, j
                break
        if row != -1:
            break

    # If all cells are covered, we're done
    if row == -1:
        return True

    # Try to place each available tile at the empty cell
    for tile_idx, (a, b) in enumerate(available_tiles):
        # Try the tile in both orientations
        for height, width in [(a, b), (b, a)]:
            if row + height <= h and col + width <= w:
                # Check if the area is empty
                valid = True
                for di in range(height):
                    for dj in range(width):
                        if grid[row+di][col+dj]:
                            valid = False
                            break
                    if not valid:
                        break

                if valid:
                    # Place the tile
                    for di in range(height):
                        for dj in range(width):
                            grid[row+di][col+dj] = True

                    new_available = available_tiles[:tile_idx] + available_tiles[tile_idx+1:]
                    if can_place_tiles(grid, new_available, h, w):
                        return True

                    # Remove the tile (backtrack)
                    for di in range(height):
                        for dj in range(width):
                            grid[row+di][col+dj] = False

    return False

n, h, w = map(int, input().split())
tiles = []
for _ in range(n):
    a, b = map(int, input().split())
    tiles.append((a, b))

# Check if the total area of tiles is sufficient
total_area = sum(a * b for a, b in tiles)
grid_area = h * w

if total_area < grid_area:
    print("No")
else:
    grid = [[False for _ in range(w)] for _ in range(h)]
    if can_place_tiles(grid, tiles, h, w):
        print("Yes")
    else:
        print("No")