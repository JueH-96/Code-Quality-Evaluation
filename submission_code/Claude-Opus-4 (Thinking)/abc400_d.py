from collections import deque

H, W = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input().strip())
A, B, C, D = map(int, input().split())
A -= 1  # Convert to 0-indexed
B -= 1
C -= 1
D -= 1

directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# 0-1 BFS
INF = float('inf')
dist = [[INF] * W for _ in range(H)]
dist[A][B] = 0
dq = deque([(A, B)])

while dq:
    r, c = dq.popleft()
    
    if dist[r][c] > dist[C][D]:
        continue
    
    # Move to adjacent road cells (cost 0)
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == '.' and dist[r][c] < dist[nr][nc]:
            dist[nr][nc] = dist[r][c]
            dq.appendleft((nr, nc))
    
    # Try kicking in each direction (cost 1)
    for dr, dc in directions:
        # Create temporary grid after kick
        temp_grid = [list(row) for row in grid]
        
        # Apply kick - turns walls into roads up to 2 cells away
        for k in range(1, 3):
            kr, kc = r + k * dr, c + k * dc
            if 0 <= kr < H and 0 <= kc < W:
                temp_grid[kr][kc] = '.'
        
        # BFS to find all cells reachable from (r, c) after the kick
        visited = [[False] * W for _ in range(H)]
        visited[r][c] = True
        q = deque([(r, c)])
        reachable = []
        
        while q:
            cr, cc = q.popleft()
            reachable.append((cr, cc))
            
            for ddr, ddc in directions:
                nnr, nnc = cr + ddr, cc + ddc
                if 0 <= nnr < H and 0 <= nnc < W and not visited[nnr][nnc] and temp_grid[nnr][nnc] == '.':
                    visited[nnr][nnc] = True
                    q.append((nnr, nnc))
        
        # Update distances for all reachable cells
        for rrr, rcc in reachable:
            if dist[r][c] + 1 < dist[rrr][rcc]:
                dist[rrr][rcc] = dist[r][c] + 1
                dq.append((rrr, rcc))

print(dist[C][D])