import sys

def solve():
    h, w = map(int, sys.stdin.readline().split())
    grid = [list(sys.stdin.readline().strip()) for _ in range(h)]
    red_cells = []
    for i in range(h):
        for j in range(w):
            if grid[i][j] == '.':
                red_cells.append((i, j))
    
    initial_grid = [row[:] for row in grid]
    
    def get_green_components(current_grid):
        rows = len(current_grid)
        cols = len(current_grid[0])
        visited = [[False for _ in range(cols)] for _ in range(rows)]
        components_count = 0
        for i in range(rows):
            for j in range(cols):
                if current_grid[i][j] == '#' and not visited[i][j]:
                    components_count += 1
                    q = [(i, j)]
                    visited[i][j] = True
                    while q:
                        r, c = q.pop(0)
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < rows and 0 <= nc < cols and current_grid[nr][nc] == '#' and not visited[nr][nc]:
                                visited[nr][nc] = True
                                q.append((nr, nc))
        return components_count
        
    component_counts = []
    for r_cell in red_cells:
        r_row, r_col = r_cell
        temp_grid = [row[:] for row in initial_grid]
        temp_grid[r_row][r_col] = '#'
        components = get_green_components(temp_grid)
        component_counts.append(components)
        
    total_components = sum(component_counts)
    num_red_cells = len(red_cells)
    
    if num_red_cells == 0:
        print(0)
        return
        
    expected_value_numerator = total_components
    expected_value_denominator = num_red_cells
    
    mod = 998244353
    
    def power(a, b):
        res = 1
        a %= mod
        while b > 0:
            if b % 2 == 1:
                res = (res * a) % mod
            a = (a * a) % mod
            b //= 2
        return res
        
    def inverse(n):
        return power(n, mod - 2)
        
    result = (expected_value_numerator * inverse(expected_value_denominator)) % mod
    print(result)

if __name__ == '__main__':
    solve()