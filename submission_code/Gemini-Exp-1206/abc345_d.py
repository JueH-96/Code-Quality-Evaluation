def solve():
    n, h, w = map(int, input().split())
    tiles = []
    for _ in range(n):
        tiles.append(list(map(int, input().split())))

    def backtrack(grid, tile_idx):
        if tile_idx == n:
            for row in grid:
                if 0 in row:
                    return False
            return True

        for i in range(h):
            for j in range(w):
                if grid[i][j] == 0:
                    
                    for _ in range(2):
                        a, b = tiles[tile_idx]
                        
                        if i + a <= h and j + b <= w:
                            valid = True
                            for r in range(i, i + a):
                                for c in range(j, j + b):
                                    if grid[r][c] != 0:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if valid:
                                new_grid = [row[:] for row in grid]
                                for r in range(i, i + a):
                                    for c in range(j, j + b):
                                        new_grid[r][c] = tile_idx + 1
                                if backtrack(new_grid, tile_idx + 1):
                                    return True
                        
                        tiles[tile_idx][0], tiles[tile_idx][1] = tiles[tile_idx][1], tiles[tile_idx][0]
                    
                    return False

        return backtrack(grid, tile_idx + 1)

    
    for i in range(1 << n):
        total_area = 0
        selected_tiles = []
        for j in range(n):
            if (i >> j) & 1:
                total_area += tiles[j][0] * tiles[j][1]
                selected_tiles.append(j)
        
        if total_area == h * w:
            
            def backtrack_selected(grid, tile_indices):
                if not tile_indices:
                    for row in grid:
                        if 0 in row:
                            return False
                    return True

                tile_idx = tile_indices[0]
                
                for i in range(h):
                    for j in range(w):
                        if grid[i][j] == 0:
                            
                            for _ in range(2):
                                a, b = tiles[tile_idx]
                                
                                if i + a <= h and j + b <= w:
                                    valid = True
                                    for r in range(i, i + a):
                                        for c in range(j, j + b):
                                            if grid[r][c] != 0:
                                                valid = False
                                                break
                                        if not valid:
                                            break
                                    if valid:
                                        new_grid = [row[:] for row in grid]
                                        for r in range(i, i + a):
                                            for c in range(j, j + b):
                                                new_grid[r][c] = tile_idx + 1
                                        if backtrack_selected(new_grid, tile_indices[1:]):
                                            return True
                                
                                tiles[tile_idx][0], tiles[tile_idx][1] = tiles[tile_idx][1], tiles[tile_idx][0]
                            
                            return False

                return backtrack_selected(grid, tile_indices[1:])
            
            if backtrack_selected([[0] * w for _ in range(h)], selected_tiles):
                print("Yes")
                return

    print("No")

solve()