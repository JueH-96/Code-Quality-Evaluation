from collections import deque

def modular_inverse(a, m):
    return pow(a, m - 2, m)

def assign_component_ids(grid, H, W):
    component_id = [[-1 for _ in range(W)] for _ in range(H)]
    current_id = 0

    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#' and component_id[i][j] == -1:
                queue = deque([(i, j)])
                component_id[i][j] = current_id

                while queue:
                    x, y = queue.popleft()
                    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] == '#' and component_id[nx][ny] == -1:
                            component_id[nx][ny] = current_id
                            queue.append((nx, ny))

                current_id += 1

    return component_id, current_id

H, W = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input())

# Identify red cells
red_cells = []
for i in range(H):
    for j in range(W):
        if grid[i][j] == '.':
            red_cells.append((i, j))

# Count the initial number of green components
component_id, num_components = assign_component_ids(grid, H, W)

total = 0
for i, j in red_cells:
    # Check adjacent cells
    adjacent_components = set()
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nx, ny = i + dx, j + dy
        if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] == '#':
            adjacent_components.add(component_id[nx][ny])

    # Calculate the number of components after repainting
    if not adjacent_components:
        # Case 1: No adjacent green cells
        components_after_repaint = num_components + 1
    else:
        # Case 2 & 3: Some adjacent green cells
        components_after_repaint = num_components - (len(adjacent_components) - 1)

    total += components_after_repaint

P = total
Q = len(red_cells)
M = 998244353

Q_inv = modular_inverse(Q, M)
R = (P * Q_inv) % M
print(R)