from collections import deque

H, W = map(int, input().split())
F = []
for i in range(H):
    row = list(map(int, input().split()))
    F.append(row)

Q = int(input())

directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

for _ in range(Q):
    A, B, Y, C, D, Z = map(int, input().split())
    # Convert to 0-indexed
    A -= 1
    B -= 1
    C -= 1
    D -= 1
    
    # 0-1 BFS
    dq = deque([(A, B, Y, 0)])  # (row, col, floor, cost)
    visited = set()
    visited.add((A, B, Y))
    
    while dq:
        r, c, floor, cost = dq.popleft()
        
        if r == C and c == D and floor == Z:
            print(cost)
            break
        
        # Try moving with stairs within the same building
        # Move up
        if floor < F[r][c]:
            if (r, c, floor + 1) not in visited:
                visited.add((r, c, floor + 1))
                dq.append((r, c, floor + 1, cost + 1))
        
        # Move down
        if floor > 1:
            if (r, c, floor - 1) not in visited:
                visited.add((r, c, floor - 1))
                dq.append((r, c, floor - 1, cost + 1))
        
        # Try moving with walkway to adjacent buildings
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and F[nr][nc] >= floor:
                if (nr, nc, floor) not in visited:
                    visited.add((nr, nc, floor))
                    dq.appendleft((nr, nc, floor, cost))