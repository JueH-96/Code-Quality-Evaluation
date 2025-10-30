# YOUR CODE HERE
def solve():
    n, h, w = map(int, input().split())
    tiles = []
    for _ in range(n):
        tiles.append(tuple(map(int, input().split())))

    def check(used_tiles):
        total_area = 0
        for i in range(n):
            if (1 << i) & used_tiles:
                total_area += tiles[i][0] * tiles[i][1]
        if total_area != h * w:
            return False

        def find_placement(grid, remaining_tiles):
            if not remaining_tiles:
                return True

            r, c = -1, -1
            for i in range(h):
                for j in range(w):
                    if grid[i][j] == 0:
                        r, c = i, j
                        break
                if r != -1:
                    break
            
            if r == -1:
                return False
            
            for tile_index in remaining_tiles:
                a, b = tiles[tile_index]
                
                # Try placing without rotation
                if r + a <= h and c + b <= w:
                    valid = True
                    for i in range(r, r + a):
                        for j in range(c, c + b):
                            if grid[i][j] != 0:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        new_grid = [row[:] for row in grid]
                        for i in range(r, r + a):
                            for j in range(c, c + b):
                                new_grid[i][j] = 1
                        new_remaining_tiles = set(remaining_tiles)
                        new_remaining_tiles.remove(tile_index)
                        if find_placement(new_grid, new_remaining_tiles):
                            return True

                # Try placing with rotation
                if r + b <= h and c + a <= w:
                    valid = True
                    for i in range(r, r + b):
                        for j in range(c, c + a):
                            if grid[i][j] != 0:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        new_grid = [row[:] for row in grid]
                        for i in range(r, r + b):
                            for j in range(c, c + a):
                                new_grid[i][j] = 1
                        new_remaining_tiles = set(remaining_tiles)
                        new_remaining_tiles.remove(tile_index)
                        if find_placement(new_grid, new_remaining_tiles):
                            return True
            return False

        used_tile_indices = []
        for i in range(n):
            if (1 << i) & used_tiles:
                used_tile_indices.append(i)
        
        grid = [[0 for _ in range(w)] for _ in range(h)]
        return find_placement(grid, set(used_tile_indices))

    for i in range(1 << n):
        if check(i):
            print("Yes")
            return

    print("No")

solve()