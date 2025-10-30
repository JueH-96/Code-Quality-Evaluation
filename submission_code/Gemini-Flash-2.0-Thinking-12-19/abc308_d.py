import collections

def solve():
    h, w = map(int, input().split())
    grid_strings = [input() for _ in range(h)]
    grid = []
    for i in range(h):
        grid.append(list(grid_strings[i]))
    
    target_sequence = "snuke"
    
    visited = [[[False] * 5 for _ in range(w)] for _ in range(h)]
    queue = collections.deque([(0, 0, 0)]) # (row, col, step_index)
    visited[0][0][0] = True
    
    while queue:
        current_row, current_col, current_step_index = queue.popleft()
        expected_char = target_sequence[current_step_index]
        actual_char = grid[current_row][current_col]
        
        if actual_char != expected_char:
            continue
            
        if current_row == h - 1 and current_col == w - 1:
            return "Yes"
            
        # Explore neighbors
        neighbors = [(current_row - 1, current_col), (current_row + 1, current_col), 
                     (current_row, current_col - 1), (current_row, current_col + 1)]
        
        for next_row, next_col in neighbors:
            if 0 <= next_row < h and 0 <= next_col < w:
                next_step_index = (current_step_index + 1) % 5
                if not visited[next_row][next_col][next_step_index]:
                    visited[next_row][next_col][next_step_index] = True
                    queue.append((next_row, next_col, next_step_index))
                    
    return "No"

if __name__ == '__main__':
    result = solve()
    print(result)