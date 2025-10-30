def count_possible_positions(H, W, N, T, grid):
    # Initialize the possible positions set with all land cells
    possible_positions = {(i, j) for i in range(H) for j in range(W) if grid[i][j] == '.'}
    
    # Define the movements
    moves = {'L': (0, -1), 'R': (0, 1), 'U': (-1, 0), 'D': (1, 0)}
    
    # Apply each move in T to the set of possible positions
    for move in T:
        # Calculate the new set of possible positions after the move
        new_positions = set()
        di, dj = moves[move]
        for i, j in possible_positions:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == '.':
                new_positions.add((ni, nj))
        possible_positions = new_positions
    
    # Return the number of possible positions
    return len(possible_positions)

# Read input from stdin
H, W, N = map(int, input().split())
T = input().strip()
grid = [input().strip() for _ in range(H)]

# Solve the problem and write the answer to stdout
print(count_possible_positions(H, W, N, T, grid))