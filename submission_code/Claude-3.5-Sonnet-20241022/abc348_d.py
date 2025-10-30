from collections import deque

def solve():
    H, W = map(int, input().split())
    grid = [list(input()) for _ in range(H)]
    N = int(input())
    medicines = []
    for _ in range(N):
        r, c, e = map(int, input().split())
        medicines.append((r-1, c-1, e))
    
    # Find start and goal
    start = None
    goal = None
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                start = (i, j)
            elif grid[i][j] == 'T':
                goal = (i, j)
    
    # BFS state: (row, col, energy, medicine_state)
    # medicine_state is a bitmask representing which medicines are used
    visited = set()
    q = deque([(start[0], start[1], 0, 0)])
    
    while q:
        r, c, energy, med_state = q.popleft()
        
        if (r, c) == goal:
            print("Yes")
            return
            
        state = (r, c, energy, med_state)
        if state in visited:
            continue
        visited.add(state)
        
        # Try using medicine at current position
        for i, (mr, mc, e) in enumerate(medicines):
            if (mr, mc) == (r, c) and not (med_state & (1 << i)):
                new_med_state = med_state | (1 << i)
                q.append((r, c, e, new_med_state))
        
        # Try moving in 4 directions if energy > 0
        if energy > 0:
            for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] != '#':
                    q.append((nr, nc, energy-1, med_state))
    
    print("No")

solve()