from collections import deque

def min_kicks(H, W, grid, A, B, C, D):
    # Convert to 0-indexed
    A -= 1
    B -= 1
    C -= 1
    D -= 1
    
    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Initialize
    min_kicks = [[float('inf') for _ in range(W)] for _ in range(H)]
    min_kicks[A][B] = 0
    
    # 0-1 BFS
    queue = deque([(A, B, 0)])  # (row, col, kicks)
    
    while queue:
        r, c, k = queue.popleft()
        
        if r == C and c == D:
            return k
        
        # If there's a better path to this cell, skip
        if min_kicks[r][c] < k:
            continue
        
        # Try moving to adjacent road cells without a kick
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == '.' and k < min_kicks[nr][nc]:
                min_kicks[nr][nc] = k
                queue.appendleft((nr, nc, k))  # Priority for no-kick moves
        
        # Try performing a front kick in each direction
        for dr, dc in directions:
            # Simulate the kick
            new_roads = []
            for step in [1, 2]:
                nr, nc = r + dr * step, c + dc * step
                if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == '#':
                    new_roads.append((nr, nc))
            
            if new_roads:  # If there are walls to kick
                # BFS to find all reachable cells after the kick
                kick_queue = deque([(r, c)])
                kick_visited = set([(r, c)])
                
                while kick_queue:
                    kr, kc = kick_queue.popleft()
                    
                    for kdr, kdc in directions:
                        knr, knc = kr + kdr, kc + kdc
                        if 0 <= knr < H and 0 <= knc < W and (knr, knc) not in kick_visited:
                            if grid[knr][knc] == '.' or (knr, knc) in new_roads:
                                kick_queue.append((knr, knc))
                                kick_visited.add((knr, knc))
                                
                                if k + 1 < min_kicks[knr][knc]:
                                    min_kicks[knr][knc] = k + 1
                                    queue.append((knr, knc, k + 1))  # Add to the back (cost 1)
    
    return min_kicks[C][D]

# Read inputs
H, W = map(int, input().split())
grid = [input() for _ in range(H)]
A, B, C, D = map(int, input().split())

# Solve
print(min_kicks(H, W, grid, A, B, C, D))