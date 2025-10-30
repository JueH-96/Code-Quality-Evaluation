def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    grid = [data[i + 2] for i in range(H)]
    
    MOD = 998244353
    
    # Directions for adjacency (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Helper function to check if a position is within the grid
    def in_bounds(x, y):
        return 0 <= x < H and 0 <= y < W
    
    # Find all connected components of green cells using DFS
    visited = [[False] * W for _ in range(H)]
    component_count = 0
    component_size = []
    
    def dfs(x, y):
        stack = [(x, y)]
        size = 0
        while stack:
            cx, cy = stack.pop()
            if visited[cx][cy]:
                continue
            visited[cx][cy] = True
            size += 1
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if in_bounds(nx, ny) and not visited[nx][ny] and grid[nx][ny] == '#':
                    stack.append((nx, ny))
        return size
    
    # Calculate initial number of components and their sizes
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#' and not visited[i][j]:
                size = dfs(i, j)
                component_count += 1
                component_size.append(size)
    
    # Count red cells and calculate the effect of turning each red cell green
    total_red_cells = 0
    expected_change = 0
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                total_red_cells += 1
                # Check how many distinct green components are adjacent to this cell
                adjacent_components = set()
                for dx, dy in directions:
                    nx, ny = i + dx, j + dy
                    if in_bounds(nx, ny) and grid[nx][ny] == '#':
                        for index in range(len(component_size)):
                            if dfs(nx, ny) == component_size[index] and index not in adjacent_components:
                                adjacent_components.add(index)
                                break
                # If we turn this red cell green:
                # - If it's adjacent to no green components, it forms a new component (+1)
                # - If it's adjacent to one or more green components, it may connect them
                change = 1 - len(adjacent_components)
                expected_change += change
    
    # Calculate expected value
    if total_red_cells == 0:
        print(0)
        return
    
    expected_value = (component_count * total_red_cells + expected_change) / total_red_cells
    
    # To find R such that R * Q ≡ P (mod MOD) where Q = total_red_cells and P = numerator of expected value
    # We need to calculate R = P * Q^(-1) mod MOD
    # Using Fermat's Little Theorem, Q^(-1) mod MOD is Q^(MOD-2) mod MOD
    P = (component_count * total_red_cells + expected_change) % MOD
    Q = total_red_cells
    Q_inv = pow(Q, MOD - 2, MOD)
    R = (P * Q_inv) % MOD
    
    print(R)

if __name__ == "__main__":
    main()