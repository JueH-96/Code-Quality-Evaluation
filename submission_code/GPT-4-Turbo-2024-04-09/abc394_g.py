import sys
import heapq

input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])

floors = []
index = 2
for _ in range(H):
    floors.append(list(map(int, data[index:index+W])))
    index += W

Q = int(data[index])
index += 1

queries = []
for _ in range(Q):
    A, B, Y, C, D, Z = map(int, data[index:index+6])
    queries.append((A-1, B-1, Y, C-1, D-1, Z))
    index += 6

# Directions for adjacency
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def bfs_min_stairs(start_r, start_c, start_floor, end_r, end_c, end_floor):
    # Priority queue: (stairs_used, current_row, current_col, current_floor)
    pq = []
    heapq.heappush(pq, (0, start_r, start_c, start_floor))
    visited = set()
    visited.add((start_r, start_c, start_floor))
    
    while pq:
        stairs_used, r, c, floor = heapq.heappop(pq)
        
        # Check if we reached the destination
        if (r, c) == (end_r, end_c):
            # Calculate the stairs needed to reach the exact floor
            return stairs_used + abs(floor - end_floor)
        
        # Explore the same building first (up and down)
        if floor > 1 and (r, c, floor-1) not in visited:
            visited.add((r, c, floor-1))
            heapq.heappush(pq, (stairs_used + 1, r, c, floor-1))
        if floor < floors[r][c] and (r, c, floor+1) not in visited:
            visited.add((r, c, floor+1))
            heapq.heappush(pq, (stairs_used + 1, r, c, floor+1))
        
        # Explore adjacent buildings via walkways
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and floors[nr][nc] >= floor and (nr, nc, floor) not in visited:
                visited.add((nr, nc, floor))
                heapq.heappush(pq, (stairs_used, nr, nc, floor))
    
    return float('inf')  # Should not reach here for valid queries

results = []
for A, B, Y, C, D, Z in queries:
    result = bfs_min_stairs(A, B, Y, C, D, Z)
    results.append(result)

for result in results:
    print(result)