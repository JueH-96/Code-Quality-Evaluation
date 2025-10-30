import heapq

H, W = map(int, input().split())
F = []
for _ in range(H):
    row = list(map(int, input().split()))
    F.append(row)

Q = int(input())
for _ in range(Q):
    A, B, Y, C, D, Z = map(int, input().split())
    A -= 1  # Convert to 0-indexed
    B -= 1
    C -= 1
    D -= 1
    
    # Dijkstra's algorithm
    pq = [(0, A, B, Y)]
    visited = set()
    
    while pq:
        cost, r, c, floor = heapq.heappop(pq)
        
        if (r, c, floor) in visited:
            continue
        visited.add((r, c, floor))
        
        if r == C and c == D and floor == Z:
            print(cost)
            break
        
        # Move up/down within the same building
        if floor < F[r][c]:
            if (r, c, floor + 1) not in visited:
                heapq.heappush(pq, (cost + 1, r, c, floor + 1))
        if floor > 1:
            if (r, c, floor - 1) not in visited:
                heapq.heappush(pq, (cost + 1, r, c, floor - 1))
        
        # Move to adjacent buildings via walkway
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and floor <= F[nr][nc]:
                if (nr, nc, floor) not in visited:
                    heapq.heappush(pq, (cost, nr, nc, floor))