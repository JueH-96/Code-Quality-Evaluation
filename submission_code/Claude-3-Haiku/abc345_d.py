from typing import List, Tuple

def can_cover_grid(N: int, H: int, W: int, tiles: List[Tuple[int, int]]) -> bool:
    def dfs(x: int, y: int, used: List[bool]) -> bool:
        if x == H and y == W:
            return True
        
        if y == W:
            return dfs(x + 1, 0, used)
        
        for i in range(N):
            if not used[i]:
                for dx, dy in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                    a, b = tiles[i]
                    if x + dx < H and y + dy < W and (dx == 0 or a >= dy) and (dy == 0 or b >= dx):
                        used[i] = True
                        if dfs(x, y + 1, used):
                            return True
                        used[i] = False
        
        return False
    
    used = [False] * N
    return dfs(0, 0, used)

# Read input
N, H, W = map(int, input().split())
tiles = []
for _ in range(N):
    a, b = map(int, input().split())
    tiles.append((a, b))

# Solve the problem
if can_cover_grid(N, H, W, tiles):
    print("Yes")
else:
    print("No")