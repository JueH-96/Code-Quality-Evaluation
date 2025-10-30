def solve():
    h, w = map(int, input().split())
    grid_str = [input() for _ in range(h)]
    grid = [list(row) for row in grid_str]
    red_cells = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == '.':
                red_cells.append((r, c))
    num_red_cells = len(red_cells)
    if num_red_cells == 0:
        
        def get_components(current_grid):
            rows = len(current_grid)
            cols = len(current_grid[0])
            visited = [[False for _ in range(cols)] for _ in range(rows)]
            component_count = 0
            for r in range(rows):
                for c in range(cols):
                    if current_grid[r][c] == '#' and not visited[r][c]:
                        component_count += 1
                        q = [(r, c)]
                        visited[r][c] = True
                        while q:
                            row, col = q.pop(0)
                            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                                nr, nc = row + dr, col + dc
                                if 0 <= nr < rows and 0 <= nc < cols and current_grid[nr][nc] == '#' and not visited[nr][nc]:
                                    visited[nr][nc] = True
                                    q.append((nr, nc))
            return component_count
            
        initial_components = get_components(grid)
        print(initial_components)
        return

    def get_initial_components_info(initial_grid):
        rows = len(initial_grid)
        cols = len(initial_grid[0])
        visited = [[False for _ in range(cols)] for _ in range(rows)]
        component_id_grid = [[0 for _ in range(cols)] for _ in range(rows)]
        component_count = 0
        for r in range(rows):
            for c in range(cols):
                if initial_grid[r][c] == '#' and not visited[r][c]:
                    component_count += 1
                    component_id = component_count
                    q = [(r, c)]
                    visited[r][c] = True
                    component_id_grid[r][c] = component_id
                    while q:
                        row, col = q.pop(0)
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = row + dr, col + dc
                            if 0 <= nr < rows and 0 <= nc < cols and initial_grid[nr][nc] == '#' and not visited[nr][nc]:
                                visited[nr][nc] = True
                                component_id_grid[nr][nc] = component_id
                                q.append((nr, nc))
        return component_count, component_id_grid

    initial_components_count, component_ids = get_initial_components_info(grid)
    sum_u_rc = 0
    for r_cell_r, r_cell_c in red_cells:
        adjacent_components = set()
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor_r, neighbor_c = r_cell_r + dr, r_cell_c + dc
            if 0 <= neighbor_r < h and 0 <= neighbor_c < w and grid[neighbor_r][neighbor_c] == '#':
                component_id = component_ids[neighbor_r][neighbor_c]
                adjacent_components.add(component_id)
        sum_u_rc += len(adjacent_components)

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
        
    def mod_inverse(n):
        return power(n, mod - 2)

    r_inv = mod_inverse(num_red_cells)
    expected_value = (initial_components_count + 1 - (sum_u_rc * r_inv) % mod) % mod
    if expected_value < 0:
        expected_value += mod
    print(expected_value)

if __name__ == '__main__':
    solve()