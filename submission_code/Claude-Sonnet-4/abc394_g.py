import heapq
from collections import defaultdict

def solve():
    H, W = map(int, input().split())
    F = []
    for i in range(H):
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
        # State: (cost, row, col, floor)
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
            
            # Move up/down floors in same building
            if floor > 1:  # Can go down
                if (r, c, floor - 1) not in visited:
                    heapq.heappush(pq, (cost + 1, r, c, floor - 1))
            if floor < F[r][c]:  # Can go up
                if (r, c, floor + 1) not in visited:
                    heapq.heappush(pq, (cost + 1, r, c, floor + 1))
            
            # Move to adjacent buildings via walkway (same floor)
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if F[nr][nc] >= floor:  # Adjacent building has this floor
                        if (nr, nc, floor) not in visited:
                            heapq.heappush(pq, (cost, nr, nc, floor))

solve()