# YOUR CODE HERE
def is_tak_code(grid, start_i, start_j):
    # Check top-left 3x3 region
    for i in range(3):
        for j in range(3):
            if grid[start_i + i][start_j + j] != '#':
                return False
    
    # Check bottom-right 3x3 region
    for i in range(3):
        for j in range(3):
            if grid[start_i + 6 + i][start_j + 6 + j] != '#':
                return False
    
    # Check adjacent cells to top-left 3x3 region
    adjacent_top_left = [
        (0, 3), (1, 3), (2, 3),  # right side
        (3, 0), (3, 1), (3, 2),  # bottom side
        (3, 3)                   # bottom-right corner
    ]
    for di, dj in adjacent_top_left:
        if grid[start_i + di][start_j + dj] != '.':
            return False
    
    # Check adjacent cells to bottom-right 3x3 region
    adjacent_bottom_right = [
        (5, 6), (5, 7), (5, 8),  # top side
        (6, 5), (7, 5), (8, 5),  # left side
        (5, 5)                   # top-left corner
    ]
    for di, dj in adjacent_bottom_right:
        if grid[start_i + 6 + di][start_j + 6 + dj] != '.':
            return False
    
    return True

def find_tak_codes(grid, N, M):
    results = []
    for i in range(N - 8):
        for j in range(M - 8):
            if is_tak_code(grid, i, j):
                results.append((i + 1, j + 1))  # Convert to 1-based index
    return results

def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N, M = map(int, data[0].split())
    grid = data[1:N+1]
    
    results = find_tak_codes(grid, N, M)
    
    for i, j in results:
        print(i, j)

main()