from collections import deque

def main():
    import sys
    data = sys.stdin.read().splitlines()
    H, W = map(int, data[0].split())
    grid = []
    for i in range(1, 1+H):
        grid.append(data[i].strip())
    
    if grid[0][0] != 's':
        print("No")
        return
        
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    snuke = "snuke"
    visited = [[[False] * 5 for _ in range(W)] for __ in range(H)]
    queue = deque()
    visited[0][0][0] = True
    queue.append((0, 0, 0))
    
    while queue:
        i, j, mod = queue.popleft()
        if i == H-1 and j == W-1:
            print("Yes")
            return
            
        next_mod = (mod + 1) % 5
        for dx, dy in directions:
            ni, nj = i + dx, j + dy
            if 0 <= ni < H and 0 <= nj < W:
                if grid[ni][nj] == snuke[next_mod]:
                    if not visited[ni][nj][next_mod]:
                        visited[ni][nj][next_mod] = True
                        queue.append((ni, nj, next_mod))
                            
    print("No")

if __name__ == "__main__":
    main()