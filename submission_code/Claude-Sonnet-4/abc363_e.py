from collections import deque

def solve():
    H, W, Y = map(int, input().split())
    A = []
    for _ in range(H):
        row = list(map(int, input().split()))
        A.append(row)
    
    # Track which sections are sunken
    sunken = [[False] * W for _ in range(H)]
    
    # Directions for adjacent cells (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def is_valid(r, c):
        return 0 <= r < H and 0 <= c < W
    
    def is_adjacent_to_sea_or_sunken(r, c):
        # Check if on border (adjacent to sea)
        if r == 0 or r == H-1 or c == 0 or c == W-1:
            return True
        
        # Check if adjacent to any sunken section
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if is_valid(nr, nc) and sunken[nr][nc]:
                return True
        
        return False
    
    total_area = H * W
    
    for year in range(1, Y + 1):
        sea_level = year
        
        # Find all sections that should sink this year
        to_sink = []
        
        # First pass: find sections that are adjacent to sea/sunken and have elevation <= sea_level
        for i in range(H):
            for j in range(W):
                if not sunken[i][j] and A[i][j] <= sea_level and is_adjacent_to_sea_or_sunken(i, j):
                    to_sink.append((i, j))
        
        # Use BFS to propagate sinking
        queue = deque(to_sink)
        newly_sunken = set(to_sink)
        
        while queue:
            r, c = queue.popleft()
            if sunken[r][c]:
                continue
                
            sunken[r][c] = True
            
            # Check adjacent cells
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if (is_valid(nr, nc) and not sunken[nr][nc] and 
                    A[nr][nc] <= sea_level and (nr, nc) not in newly_sunken):
                    queue.append((nr, nc))
                    newly_sunken.add((nr, nc))
        
        # Count remaining area
        remaining_area = 0
        for i in range(H):
            for j in range(W):
                if not sunken[i][j]:
                    remaining_area += 1
        
        print(remaining_area)

solve()