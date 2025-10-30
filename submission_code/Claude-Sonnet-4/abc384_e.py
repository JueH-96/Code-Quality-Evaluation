import heapq

def solve():
    H, W, X = map(int, input().split())
    P, Q = map(int, input().split())
    P -= 1  # Convert to 0-indexed
    Q -= 1
    
    grid = []
    for _ in range(H):
        row = list(map(int, input().split()))
        grid.append(row)
    
    # Directions for adjacent cells (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Start with Takahashi's initial strength
    current_strength = grid[P][Q]
    absorbed = [[False] * W for _ in range(H)]
    absorbed[P][Q] = True
    
    # Use a max heap to always try to absorb the strongest possible slime
    # We'll use negative values since Python has min heap
    heap = []
    
    # Add initial adjacent cells to heap
    for dr, dc in directions:
        nr, nc = P + dr, Q + dc
        if 0 <= nr < H and 0 <= nc < W and not absorbed[nr][nc]:
            heapq.heappush(heap, (-grid[nr][nc], nr, nc))
    
    while heap:
        # Get the strongest available slime
        neg_strength, r, c = heapq.heappop(heap)
        strength = -neg_strength
        
        # Skip if already absorbed
        if absorbed[r][c]:
            continue
        
        # Check if this slime can be absorbed
        if strength * X >= current_strength:
            continue
        
        # Absorb the slime
        current_strength += strength
        absorbed[r][c] = True
        
        # Add newly adjacent cells to heap
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not absorbed[nr][nc]:
                heapq.heappush(heap, (-grid[nr][nc], nr, nc))
    
    print(current_strength)

solve()