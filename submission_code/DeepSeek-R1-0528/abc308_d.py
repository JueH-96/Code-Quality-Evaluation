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
        
    visited = [[[False] * 5 for _ in range(W)] for __ in range(H)]
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    queue = deque()
    queue.append((0, 0, 0))
    visited[0][0][0] = True
    
    while queue:
        i, j, mod = queue.popleft()
        if i == H-1 and j == W-1:
            print("Yes")
            return
            
        next_mod = (mod + 1) % 5
        next_letter = "snuke"[next_mod]
        
        for dx, dy in directions:
            ni, nj = i + dx, j + dy
            if 0 <= ni < H and 0 <= nj < W:
                if grid[ni][nj] == next_letter and not visited[ni][nj][next_mod]:
                    visited[ni][nj][next_mod] = True
                    queue.append((ni, nj, next_mod))
                    
    print("No")

if __name__ == '__main__':
    main()