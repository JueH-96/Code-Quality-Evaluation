import sys

sys.setrecursionlimit(100000)

def main():
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it)); H = int(next(it)); W = int(next(it))
    tiles = []
    total_area = 0
    for i in range(N):
        a = int(next(it)); b = int(next(it))
        tiles.append((a, b))
        total_area += a * b

    if total_area < H * W:
        print("No")
        return

    grid = [[False] * W for _ in range(H)]
    
    if dfs(0, 0, grid, tiles, N, H, W):
        print("Yes")
    else:
        print("No")

def dfs(used_mask, covered_count, grid, tiles, N, H, W):
    if covered_count == H * W:
        return True
        
    x, y = -1, -1
    for i in range(H):
        for j in range(W):
            if not grid[i][j]:
                x, y = i, j
                break
        if x != -1:
            break
    if x == -1:
        return True
        
    remaining_area = H * W - covered_count

    for i in range(N):
        if used_mask >> i & 1:
            continue
        a0, b0 = tiles[i]
        area_i = a0 * b0
        if area_i > remaining_area:
            continue
        orientations = set()
        orientations.add((a0, b0))
        orientations.add((b0, a0))
        for (a, b) in orientations:
            if x + a > H or y + b > W:
                continue
            valid = True
            for ii in range(x, x + a):
                for jj in range(y, y + b):
                    if grid[ii][jj]:
                        valid = False
                        break
                if not valid:
                    break
                    
            if not valid:
                continue
                
            for ii in range(x, x + a):
                for jj in range(y, y + b):
                    grid[ii][jj] = True
                    
            if dfs(used_mask | (1 << i), covered_count + a * b, grid, tiles, N, H, W):
                return True
                
            for ii in range(x, x + a):
                for jj in range(y, y + b):
                    grid[ii][jj] = False
                    
    return False

if __name__ == "__main__":
    main()