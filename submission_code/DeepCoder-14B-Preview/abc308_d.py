from collections import deque

def main():
    H, W = map(int, input().split())
    grid = [input().strip() for _ in range(H)]
    
    if grid[0][0] != 's':
        print("No")
        return
    
    target = (H-1, W-1)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    snuke = 'snuke'
    
    visited = [[[False for _ in range(5)] for __ in range(W)] for ___ in range(H)]
    queue = deque()
    queue.append((0, 0, 0))
    visited[0][0][0] = True
    
    found = False
    
    while queue:
        i, j, mod = queue.popleft()
        
        if (i, j) == target:
            found = True
            break
        
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                new_mod = (mod + 1) % 5
                if grid[ni][nj] == snuke[new_mod]:
                    if not visited[ni][nj][new_mod]:
                        visited[ni][nj][new_mod] = True
                        queue.append((ni, nj, new_mod))
    
    print("Yes" if found else "No")

if __name__ == "__main__":
    main()