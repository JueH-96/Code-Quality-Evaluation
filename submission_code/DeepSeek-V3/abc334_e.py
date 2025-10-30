import sys
from collections import deque

MOD = 998244353

def inv(x, mod):
    return pow(x, mod-2, mod)

def main():
    H, W = map(int, sys.stdin.readline().split())
    grid = [list(sys.stdin.readline().strip()) for _ in range(H)]
    
    # Count the number of red cells
    red_count = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                red_count += 1
    
    if red_count == 0:
        print(0)
        return
    
    # Initialize the grid with the original green cells
    # We need to find the connected components in the original grid
    # and then for each red cell, determine how its addition affects the number of connected components
    
    # First, find all connected components in the original grid
    visited = [[False for _ in range(W)] for _ in range(H)]
    original_components = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#' and not visited[i][j]:
                original_components += 1
                q = deque()
                q.append((i, j))
                visited[i][j] = True
                while q:
                    x, y = q.popleft()
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < H and 0 <= ny < W:
                            if grid[nx][ny] == '#' and not visited[nx][ny]:
                                visited[nx][ny] = True
                                q.append((nx, ny))
    
    # Now, for each red cell, determine how its addition affects the number of connected components
    # We need to count the number of adjacent green cells and see if they are in different components
    # To do this, we need to precompute the component number for each green cell
    component = [[-1 for _ in range(W)] for _ in range(H)]
    component_id = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#' and component[i][j] == -1:
                q = deque()
                q.append((i, j))
                component[i][j] = component_id
                while q:
                    x, y = q.popleft()
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < H and 0 <= ny < W:
                            if grid[nx][ny] == '#' and component[nx][ny] == -1:
                                component[nx][ny] = component_id
                                q.append((nx, ny))
                component_id += 1
    
    # Now, for each red cell, count the number of unique adjacent components
    # The change in the number of connected components is 1 - (number of unique adjacent components)
    total = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                adjacent_components = set()
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = i + dx, j + dy
                    if 0 <= nx < H and 0 <= ny < W:
                        if grid[nx][ny] == '#':
                            adjacent_components.add(component[nx][ny])
                if not adjacent_components:
                    # No adjacent green cells, adding this cell creates a new component
                    total += original_components + 1
                else:
                    # The number of connected components decreases by (number of adjacent components - 1)
                    total += original_components - (len(adjacent_components) - 1)
    
    # The expected value is total / red_count
    # Since we need to compute it modulo MOD, we need to find the modular inverse of red_count
    inv_red_count = inv(red_count, MOD)
    expected = (total * inv_red_count) % MOD
    print(expected)

if __name__ == "__main__":
    main()