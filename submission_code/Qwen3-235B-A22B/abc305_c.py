import sys

def main():
    h, w = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(h)]
    
    # Collect all positions of '#'
    positions = []
    for i in range(h):
        for j in range(w):
            if grid[i][j] == '#':
                positions.append((i, j))
    
    # Determine the minimal enclosing rectangle (0-based)
    min_row = min(p[0] for p in positions)
    max_row = max(p[0] for p in positions)
    min_col = min(p[1] for p in positions)
    max_col = max(p[1] for p in positions)
    
    # Find the eaten cell within this rectangle
    eaten = []
    for i in range(min_row, max_row + 1):
        for j in range(min_col, max_col + 1):
            if grid[i][j] == '.':
                eaten.append((i + 1, j + 1))  # Convert to 1-based
    
    print(eaten[0][0], eaten[0][1])

if __name__ == "__main__":
    main()