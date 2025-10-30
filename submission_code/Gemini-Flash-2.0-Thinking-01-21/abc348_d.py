import collections

def solve():
    h, w = map(int, input().split())
    grid = [list(input()) for _ in range(h)]
    n_medicines = int(input())
    medicines = []
    for _ in range(n_medicines):
        r, c, e = map(int, input().split())
        medicines.append(((r - 1, c - 1), e))
    
    start_pos = None
    goal_pos = None
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 'S':
                start_pos = (r, c)
                grid[r][c] = '.'
            elif grid[r][c] == 'T':
                goal_pos = (r, c)
                grid[r][c] = '.'
                
    medicine_locations = {}
    for i in range(n_medicines):
        location, energy = medicines[i]
        medicine_locations[location] = (energy, i)
        
    start_r, start_c = start_pos
    goal_r, goal_c = goal_pos
    
    max_energy = h * w
    visited_states = set()
    initial_state = (start_r, start_c, 0, 0) # (row, column, energy, used_medicines_mask)
    queue = collections.deque([initial_state])
    visited_states.add(initial_state)
    
    while queue:
        current_r, current_c, current_energy, used_mask = queue.popleft()
        
        if (current_r, current_c) == goal_pos:
            return "Yes"
            
        # Try to use medicine
        if (current_r, current_c) in medicine_locations:
            medicine_energy, medicine_index = medicine_locations[(current_r, current_c)]
            if not (used_mask & (1 << medicine_index)):
                next_energy = medicine_energy
                next_mask = used_mask | (1 << medicine_index)
                next_state = (current_r, current_c, next_energy, next_mask)
                if next_energy <= max_energy and next_state not in visited_states:
                    visited_states.add(next_state)
                    queue.append(next_state)
                    
        # Try to move to neighbors
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            next_r, next_c = current_r + dr, current_c + dc
            if 0 <= next_r < h and 0 <= next_c < w and grid[next_r][next_c] != '#' and current_energy > 0:
                next_energy = current_energy - 1
                next_mask = used_mask
                next_state = (next_r, next_c, next_energy, next_mask)
                if next_state not in visited_states:
                    visited_states.add(next_state)
                    queue.append(next_state)
                    
    return "No"

if __name__ == '__main__':
    result = solve()
    print(result)