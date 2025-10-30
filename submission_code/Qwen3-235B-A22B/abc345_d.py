import sys

def main():
    N, H, W = map(int, sys.stdin.readline().split())
    tiles = []
    for _ in range(N):
        a, b = map(int, sys.stdin.readline().split())
        tiles.append((a, b))
    
    # Function to check if the given rectangles can tile the grid
    def can_place(rects):
        grid = [[False for _ in range(W)] for _ in range(H)]
        used = [False] * len(rects)
        
        def dfs():
            # Find first empty cell
            x, y = -1, -1
            for i in range(H):
                for j in range(W):
                    if not grid[i][j]:
                        x, y = i, j
                        break
                if x != -1:
                    break
            if x == -1:
                return True  # All cells filled
            
            for idx in range(len(rects)):
                if not used[idx]:
                    a, b = rects[idx]
                    # Try both orientations
                    orientations = set()
                    orientations.add((a, b))
                    orientations.add((b, a))
                    
                    for h, w in orientations:
                        if x + h > H or y + w > W:
                            continue
                        # Check if all cells are free
                        valid = True
                        for i in range(h):
                            for j in range(w):
                                if grid[x + i][j + y]:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            # Place the tile
                            for i in range(h):
                                for j in range(w):
                                    grid[x + i][j + y] = True
                            used[idx] = True
                            if dfs():
                                return True
                            # Unplace the tile
                            for i in range(h):
                                for j in range(w):
                                    grid[x + i][j + y] = False
                            used[idx] = False
            return False
        
        return dfs()
    
    # Check all possible subsets
    for mask in range(1, 1 << N):
        subset = []
        total_area = 0
        for i in range(N):
            if (mask >> i) & 1:
                a, b = tiles[i]
                subset.append((a, b))
                total_area += a * b
        if total_area != H * W:
            continue
        
        # Sort rectangles to optimize backtracking
        rects = sorted(subset, key=lambda x: (-(x[0] * x[1]), -max(x[0], x[1])))
        if can_place(rects):
            print("Yes")
            return
    
    print("No")

if __name__ == "__main__":
    main()