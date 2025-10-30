import collections

def solve():
    n = int(input())
    grid_lines = [input() for _ in range(n)]
    grid = [list(line) for line in grid_lines]
    player_positions = []
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 'P':
                player_positions.append((i, j))
                grid[i][j] = '.'
                
    start_pos1 = player_positions[0]
    start_pos2 = player_positions[1]
    initial_state = (start_pos1, start_pos2)
    
    queue = collections.deque([initial_state])
    visited_states = {initial_state}
    moves_count = {initial_state: 0}
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # up, down, left, right
    
    while queue:
        current_state = queue.popleft()
        pos1, pos2 = current_state
        r1, c1 = pos1
        r2, c2 = pos2
        
        if pos1 == pos2:
            return moves_count[current_state]
            
        for dr, dc in directions:
            next_r1, next_c1 = r1 + dr, c1 + dc
            next_r2, next_c2 = r2 + dr, c2 + dc
            
            next_pos1 = (r1, c1)
            if 0 <= next_r1 < n and 0 <= next_c1 < n and grid[next_r1][next_c1] != '#':
                next_pos1 = (next_r1, next_c1)
                
            next_pos2 = (r2, c2)
            if 0 <= next_r2 < n and 0 <= next_c2 < n and grid[next_r2][next_c2] != '#':
                next_pos2 = (next_r2, next_c2)
                
            next_state = (next_pos1, next_pos2)
            if next_state not in visited_states:
                visited_states.add(next_state)
                moves_count[next_state] = moves_count[current_state] + 1
                queue.append(next_state)
                
    return -1

if __name__ == '__main__':
    result = solve()
    print(result)