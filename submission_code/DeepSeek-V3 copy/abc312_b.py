# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    grid = data[2:N+2]
    
    # Define the regions that must be black and white
    # For the top-left 3x3 and bottom-right 3x3 regions
    black_positions = []
    for i in range(3):
        for j in range(3):
            black_positions.append((i, j))
            black_positions.append((i+6, j+6))
    
    # Define the positions that must be white
    # These are the cells adjacent to the black regions
    white_positions = []
    for i in range(3):
        for j in range(3):
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < 9 and 0 <= nj < 9:
                        if (ni, nj) not in black_positions:
                            white_positions.append((ni, nj))
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + 6 + di, j + 6 + dj
                    if 0 <= ni < 9 and 0 <= nj < 9:
                        if (ni, nj) not in black_positions:
                            white_positions.append((ni, nj))
    
    # Remove duplicates
    white_positions = list(set(white_positions))
    
    # Now, for each possible 9x9 region in the grid, check if it satisfies the conditions
    result = []
    for i in range(N - 8):
        for j in range(M - 8):
            # Check all black positions
            valid = True
            for (di, dj) in black_positions:
                if grid[i + di][j + dj] != '#':
                    valid = False
                    break
            if not valid:
                continue
            # Check all white positions
            for (di, dj) in white_positions:
                if grid[i + di][j + dj] != '.':
                    valid = False
                    break
            if valid:
                result.append((i+1, j+1))
    
    # Sort the result
    result.sort()
    
    # Output the result
    for (i, j) in result:
        print(i, j)

if __name__ == "__main__":
    main()