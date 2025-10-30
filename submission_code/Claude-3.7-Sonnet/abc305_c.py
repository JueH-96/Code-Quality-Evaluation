# YOUR CODE HERE
def solve():
    H, W = map(int, input().split())
    grid = []
    for _ in range(H):
        grid.append(input())
    
    cookie_cells = []
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                cookie_cells.append((i, j))
    
    # Calculate the smallest rectangle that contains all cookies
    min_row = min(i for i, j in cookie_cells)
    max_row = max(i for i, j in cookie_cells)
    min_col = min(j for i, j in cookie_cells)
    max_col = max(j for i, j in cookie_cells)
    
    # Check each square within the rectangle
    for i in range(min_row, max_row + 1):
        for j in range(min_col, max_col + 1):
            if grid[i][j] == '.':
                return i + 1, j + 1  # +1 because the problem uses 1-indexing
    
    # If no missing cookie is found (which shouldn't happen according to the problem)
    return -1, -1

i, j = solve()
print(f"{i} {j}")