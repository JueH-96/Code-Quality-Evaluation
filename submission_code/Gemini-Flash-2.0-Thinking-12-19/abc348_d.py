import collections

def solve():
    h, w = map(int, input().split())
    grid = [list(input()) for _ in range(h)]
    n = int(input())
    medicines = []
    for _ in range(n):
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
                
    start_r, start_c = start_pos
    goal_r, goal_c = goal_pos
    
    medicine_locations = collections.defaultdict(list)
    for i in range(n):
        location, energy = medicines[i]
        medicine_locations[location].append((energy, i))
        
    initial_state = (start_pos, tuple()) # position, used_medicine_indices
    queue = collections.deque([initial_state])
    max_energy_at_state = {initial_state: 0}
    
    while queue:
        current_state = queue.popleft()
        current_pos, used_medicine_indices = current_state
        current_energy = max_energy_at_state[current_state]
        
        if current_pos == goal_pos:
            return "Yes"
            
        # Move actions
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            next_r, next_c = current_pos[0] + dr, current_pos[1] + dc
            if 0 <= next_r < h and 0 <= next_c < w and grid[next_r][next_c] != '#' and current_energy > 0:
                next_pos = (next_r, next_c)
                next_energy = current_energy - 1
                next_state = (next_pos, used_medicine_indices)
                if next_state not in max_energy_at_state or next_energy > max_energy_at_state[next_state]:
                    max_energy_at_state[next_state] = next_energy
                    queue.append(next_state)
                    
        # Medicine actions
        if current_pos in medicine_locations:
            for energy, medicine_index in medicine_locations[current_pos]:
                if medicine_index not in used_medicine_indices:
                    next_energy = energy
                    next_used_medicine_indices = tuple(sorted(list(used_medicine_indices) + [medicine_index]))
                    next_state = (current_pos, next_used_medicine_indices)
                    if next_state not in max_energy_at_state or next_energy > max_energy_at_state[next_state]:
                        max_energy_at_state[next_state] = next_energy
                        queue.append(next_state)
                        
    return "No"

if __name__ == '__main__':
    result = solve()
    print(result)