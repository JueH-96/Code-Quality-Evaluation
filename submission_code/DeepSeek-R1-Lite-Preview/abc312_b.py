import sys

def main():
    data = sys.stdin.read().splitlines()
    N_M = data[0].split()
    N = int(N_M[0])
    M = int(N_M[1])
    grid = data[1:N+1]
    
    # Define the adjacent cells relative to the top-left corner (i,j)
    adjacent_cells = [
        (0, 3), (1, 3), (2, 3),
        (3, 0), (3, 1), (3, 2), (3, 3),
        (5, 5), (5, 6), (5, 7), (5, 8),
        (6, 5), (7, 5), (8, 5)
    ]
    
    results = []
    
    # Iterate over all possible top-left positions of 9x9 regions
    for i in range(N - 8):
        for j in range(M - 8):
            valid = True
            
            # Check top-left 3x3 region
            for x in range(i, i + 3):
                for y in range(j, j + 3):
                    if grid[x][y] != '#':
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Check bottom-right 3x3 region
            for x in range(i + 6, i + 9):
                for y in range(j + 6, j + 9):
                    if grid[x][y] != '#':
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Check adjacent cells
            for dx, dy in adjacent_cells:
                if grid[i + dx][j + dy] != '.':
                    valid = False
                    break
            if not valid:
                continue
            
            # If all conditions are satisfied, record the position (1-based indexing)
            results.append((i + 1, j + 1))
    
    # Print the results in lexicographical order
    for pos in results:
        print(pos[0], pos[1])

if __name__ == "__main__":
    main()