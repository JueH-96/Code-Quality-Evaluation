import sys

def can_tile(grid, tiles):
    if len(tiles) == 0:
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if not grid[i][j]:
                    return False
        return True

    r, c = -1, -1
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if not grid[i][j]:
                r, c = i, j
                break
        if r != -1:
            break
    if r == -1:
        return False

    for idx in range(len(tiles)):
        a, b = tiles[idx]
        orientations = []
        orientations.append((a, b))
        if a != b:
            orientations.append((b, a))
        
        for (h_tile, w_tile) in orientations:
            if r + h_tile > len(grid) or c + w_tile > len(grid[0]):
                continue
            
            valid = True
            for i2 in range(r, r + h_tile):
                for j2 in range(c, c + w_tile):
                    if grid[i2][j2]:
                        valid = False
                        break
                if not valid:
                    break
                    
            if not valid:
                continue
                
            for i2 in range(r, r + h_tile):
                for j2 in range(c, c + w_tile):
                    grid[i2][j2] = True
                    
            new_tiles = tiles[:idx] + tiles[idx+1:]
            if can_tile(grid, new_tiles):
                return True
                
            for i2 in range(r, r + h_tile):
                for j2 in range(c, c + w_tile):
                    grid[i2][j2] = False
                    
    return False

def main():
    data = sys.stdin.read().split()
    if not data:
        print("No")
        return
        
    n = int(data[0])
    H = int(data[1])
    W = int(data[2])
    tiles = []
    index = 3
    for i in range(n):
        a = int(data[index]); b = int(data[index+1]); index += 2
        tiles.append((a, b))
        
    total_area = H * W
    min_grid = min(H, W)
    max_grid = max(H, W)
    
    for mask in range(1 << n):
        area_sum = 0
        skip_subset = False
        selected = []
        for i in range(n):
            if mask & (1 << i):
                a, b = tiles[i]
                m_val = min(a, b)
                M_val = max(a, b)
                if m_val > min_grid or M_val > max_grid:
                    skip_subset = True
                area_sum += a * b
                selected.append(tiles[i])
                
        if skip_subset:
            continue
        if area_sum != total_area:
            continue
            
        grid = [[False] * W for _ in range(H)]
        if can_tile(grid, selected):
            print("Yes")
            return
            
    print("No")

if __name__ == "__main__":
    main()