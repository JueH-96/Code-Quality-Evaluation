from collections import deque

def can_reach_goal():
    H, W = map(int, input().split())
    grid = [input() for _ in range(H)]
    N = int(input())
    
    # Store medicine info
    medicines = []
    medicine_positions = {}
    for i in range(N):
        r, c, e = map(int, input().split())
        medicines.append((r-1, c-1, e))  # Convert to 0-indexed
        medicine_positions[(r-1, c-1)] = i
    
    # Find the start and goal positions
    start, goal = None, None
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                start = (i, j)
            elif grid[i][j] == 'T':
                goal = (i, j)
    
    # Check if start and goal are the same
    if start == goal:
        return "Yes"
    
    # BFS
    queue = deque([(start[0], start[1], 0, 0)])  # (row, col, energy, used_medicines_bitset)
    visited = set([(start[0], start[1], 0, 0)])
    
    while queue:
        r, c, energy, used_meds = queue.popleft()
        
        # Check if Takahashi reached the goal
        if (r, c) == goal:
            return "Yes"
        
        # Use medicine at the current position
        if (r, c) in medicine_positions:
            med_idx = medicine_positions[(r, c)]
            if not (used_meds & (1 << med_idx)):  # Check if medicine hasn't been used
                new_energy = medicines[med_idx][2]
                new_used_meds = used_meds | (1 << med_idx)  # Set the bit for the used medicine
                if (r, c, new_energy, new_used_meds) not in visited:
                    queue.append((r, c, new_energy, new_used_meds))
                    visited.add((r, c, new_energy, new_used_meds))
        
        # Move to adjacent cells if energy > 0
        if energy > 0:
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] != '#' and (nr, nc, energy - 1, used_meds) not in visited:
                    queue.append((nr, nc, energy - 1, used_meds))
                    visited.add((nr, nc, energy - 1, used_meds))
    
    return "No"

print(can_reach_goal())