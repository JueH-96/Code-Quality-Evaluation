import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    H = int(data[idx]); idx += 1
    W = int(data[idx]); idx += 1
    X = int(data[idx]); idx += 1
    P = int(data[idx]); idx += 1
    Q = int(data[idx]); idx += 1
    
    # Convert to zero-based indexing
    P -= 1
    Q -= 1
    
    grid = []
    for _ in range(H):
        row = list(map(int, data[idx:idx+W]))
        grid.append(row)
        idx += W
    
    # Directions: up, down, left, right
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    
    visited = [[False for _ in range(W)] for _ in range(H)]
    visited[P][Q] = True
    current_strength = grid[P][Q]
    
    # Max-heap: use negative values to simulate max-heap
    heap = []
    
    # Initial adjacent slimes
    for dr, dc in directions:
        nr, nc = P + dr, Q + dc
        if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc]:
            if grid[nr][nc] * X < current_strength:
                heapq.heappush(heap, (-grid[nr][nc], nr, nc))
    
    while heap:
        neg_strength, r, c = heapq.heappop(heap)
        strength = -neg_strength
        if not visited[r][c] and strength * X < current_strength:
            visited[r][c] = True
            current_strength += strength
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc]:
                    if grid[nr][nc] * X < current_strength:
                        heapq.heappush(heap, (-grid[nr][nc], nr, nc))
    
    print(current_strength)

if __name__ == '__main__':
    main()