import collections

def solve():
    n = int(input())
    grid_str = [input() for _ in range(n)]
    grid = [list(row) for row in grid_str]
    player_positions = []
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 'P':
                player_positions.append((i, j))
                grid[i][j] = '.' # Treat initial player positions as empty cells for movement purposes

    start_pos1 = player_positions[0]
    start_pos2 = player_positions[1]
    initial_state = tuple(sorted([start_pos1, start_pos2]))
    
    queue = collections.deque([(initial_state, 0)])
    visited_states = {initial_state}
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # up, down, left, right
    
    while queue:
        current_state, moves = queue.popleft()
        pos1, pos2 = current_state
        if pos1 == pos2:
            return moves
            
        for dr, dc in directions:
            next_pos1_row = pos1[0] + dr
            next_pos1_col = pos1[1] + dc
            next_pos2_row = pos2[0] + dr
            next_pos2_col = pos2[1] + dc
            
            next_pos1 = pos1
            if 0 <= next_pos1_row < n and 0 <= next_pos1_col < n and grid[next_pos1_row][next_pos1_col] != '#':
                next_pos1 = (next_pos1_row, next_pos1_col)
                
            next_pos2 = pos2
            if 0 <= next_pos2_row < n and 0 <= next_pos2_col < n and grid[next_pos2_row][next_pos2_col] != '#':
                next_pos2 = (next_pos2_row, next_pos2_col)
                
            next_state = tuple(sorted([next_pos1, next_pos2]))
            if next_state not in visited_states:
                visited_states.add(next_state)
                queue.append((next_state, moves + 1))
                
    return -1

if __name__ == '__main__':
    result = solve()
    print(result)