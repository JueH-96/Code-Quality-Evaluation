import sys
from collections import deque

# Read input
data = sys.stdin.read().split()
index = 0
H = int(data[index])
index += 1
W = int(data[index])
index += 1
grid = []
for _ in range(H):
    row = data[index]
    index += 1
    grid.append(row)

# Helper functions
def in_bound(i, j):
    return 0 <= i < H and 0 <= j < W

def is_dot(i, j):
    return in_bound(i, j) and grid[i][j] == '.'

def is_free(i, j):
    if not is_dot(i, j):
        return False
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ni = i + di
        nj = j + dj
        if in_bound(ni, nj) and grid[ni][nj] == '#':
            return False
    return True

# Initialize visited grid and max degree
visited = [[False for _ in range(W)] for _ in range(H)]
max_deg = 0

# Iterate through all cells
for i in range(H):
    for j in range(W):
        if is_dot(i, j):
            if is_free(i, j) and not visited[i][j]:
                # Perform BFS to find the connected component of free cells
                queue = deque()
                queue.append((i, j))
                visited[i][j] = True
                comp_cells = [(i, j)]  # List to store cells in the component
                
                # BFS to find all cells in the component
                while queue:
                    ci, cj = queue.popleft()
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni = ci + di
                        nj = cj + dj
                        if in_bound(ni, nj) and is_free(ni, nj) and not visited[ni][nj]:
                            queue.append((ni, nj))
                            visited[ni][nj] = True
                            comp_cells.append((ni, nj))
                
                # Calculate component size
                comp_size = len(comp_cells)
                
                # Find adjacent cells that are '.' and not free
                adj_set = set()
                for cell in comp_cells:
                    ci, cj = cell
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni = ci + di
                        nj = cj + dj
                        if in_bound(ni, nj) and is_dot(ni, nj) and not is_free(ni, nj):
                            adj_set.add((ni, nj))
                num_adj = len(adj_set)
                
                # Calculate degree of freedom for this component
                degree = comp_size + num_adj
                max_deg = max(max_deg, degree)
            else:
                # If the cell is not free (and is dot), degree is 1
                if not is_free(i, j):
                    max_deg = max(max_deg, 1)

# Output the maximum degree of freedom
print(max_deg)