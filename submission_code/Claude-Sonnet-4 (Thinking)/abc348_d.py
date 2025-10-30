from collections import deque

def solve():
    H, W = map(int, input().split())
    grid = []
    start_pos = None
    goal_pos = None
    
    for i in range(H):
        row = input().strip()
        grid.append(row)
        for j in range(W):
            if row[j] == 'S':
                start_pos = (i, j)
            elif row[j] == 'T':
                goal_pos = (i, j)
    
    N = int(input())
    medicine_map = {}
    
    for i in range(N):
        r, c, e = map(int, input().split())
        medicine_map[(r-1, c-1)] = (i, e)  # Convert to 0-indexed
    
    # BFS with state (row, col, used_medicines_bitmask)
    queue = deque([(start_pos[0], start_pos[1], 0, 0)])  # (row, col, used_mask, energy)
    best_energy = {}  # (row, col, used_mask) -> max_energy_seen
    
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    while queue:
        row, col, used_mask, energy = queue.popleft()
        
        # Check if we reached the goal
        if (row, col) == goal_pos:
            print("Yes")
            return
        
        state = (row, col, used_mask)
        if state in best_energy and best_energy[state] >= energy:
            continue
        best_energy[state] = energy
        
        # Try using medicine at current position
        if (row, col) in medicine_map:
            med_idx, med_energy = medicine_map[(row, col)]
            if not (used_mask & (1 << med_idx)):  # Medicine not used yet
                new_mask = used_mask | (1 << med_idx)
                new_state = (row, col, new_mask)
                if new_state not in best_energy or best_energy[new_state] < med_energy:
                    queue.append((row, col, new_mask, med_energy))
        
        # Try moving to adjacent cells
        if energy > 0:
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < H and 0 <= new_col < W and grid[new_row][new_col] != '#':
                    new_energy = energy - 1
                    new_state = (new_row, new_col, used_mask)
                    if new_state not in best_energy or best_energy[new_state] < new_energy:
                        queue.append((new_row, new_col, used_mask, new_energy))
    
    print("No")

solve()