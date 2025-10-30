# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    N = int(data[2])
    T = data[3]
    S = data[4:]
    
    # Initialize the grid
    grid = [list(row) for row in S]
    
    # Initialize the possible positions
    possible_positions = set()
    
    # Find all land cells
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                possible_positions.add((i, j))
    
    # Apply the moves to each possible starting position
    for move in T:
        new_positions = set()
        for i, j in possible_positions:
            if move == 'L' and j > 0 and grid[i][j-1] == '.':
                new_positions.add((i, j-1))
            elif move == 'R' and j < W-1 and grid[i][j+1] == '.':
                new_positions.add((i, j+1))
            elif move == 'U' and i > 0 and grid[i-1][j] == '.':
                new_positions.add((i-1, j))
            elif move == 'D' and i < H-1 and grid[i+1][j] == '.':
                new_positions.add((i+1, j))
        possible_positions = new_positions
    
    # Print the number of possible positions
    print(len(possible_positions))

if __name__ == "__main__":
    main()