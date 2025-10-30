import collections

def solve():
    h, w = map(int, input().split())
    s = [input() for _ in range(h)]
    grid = [[1 if char == '#' else 0 for char in row] for row in s]
    max_degree_of_freedom = 0
    
    for start_row in range(h):
        for start_col in range(w):
            if grid[start_row][start_col] == 0:
                reachable_cells = set()
                q = collections.deque([(start_row, start_col)])
                visited = [[False for _ in range(w)] for _ in range(h)]
                visited[start_row][start_col] = True
                reachable_cells.add((start_row, start_col))
                
                while q:
                    r, c = q.popleft()
                    is_safe = True
                    adjacent_cells_to_check = []
                    if r > 0: adjacent_cells_to_check.append((r-1, c))
                    if r < h - 1: adjacent_cells_to_check.append((r+1, c))
                    if c > 0: adjacent_cells_to_check.append((r, c-1))
                    if c < w - 1: adjacent_cells_to_check.append((r, c+1))
                    
                    for adj_r, adj_c in adjacent_cells_to_check:
                        if grid[adj_r][adj_c] == 1:
                            is_safe = False
                            break
                            
                    if is_safe:
                        possible_moves = []
                        if r > 0: possible_moves.append((r-1, c))
                        if r < h - 1: possible_moves.append((r+1, c))
                        if c > 0: possible_moves.append((r, c-1))
                        if c < w - 1: possible_moves.append((r, c+1))
                        
                        for next_r, next_c in possible_moves:
                            if 0 <= next_r < h and 0 <= next_c < w and grid[next_r][next_c] == 0 and not visited[next_r][next_c]:
                                visited[next_r][next_c] = True
                                reachable_cells.add((next_r, next_c))
                                q.append((next_r, next_c))
                                
                degree_of_freedom = len(reachable_cells)
                max_degree_of_freedom = max(max_degree_of_freedom, degree_of_freedom)
                
    print(max_degree_of_freedom)

if __name__ == '__main__':
    solve()