# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    S_i = int(data[2]) - 1  # Convert to 0-based index
    S_j = int(data[3]) - 1  # Convert to 0-based index
    
    grid = []
    index = 4
    for _ in range(H):
        grid.append(data[index])
        index += 1
    
    X = data[index]
    
    # Initial position
    current_i = S_i
    current_j = S_j
    
    # Process each character in X
    for move in X:
        if move == 'L':
            if current_j > 0 and grid[current_i][current_j - 1] == '.':
                current_j -= 1
        elif move == 'R':
            if current_j < W - 1 and grid[current_i][current_j + 1] == '.':
                current_j += 1
        elif move == 'U':
            if current_i > 0 and grid[current_i - 1][current_j] == '.':
                current_i -= 1
        elif move == 'D':
            if current_i < H - 1 and grid[current_i + 1][current_j] == '.':
                current_i += 1
    
    # Output the final position (convert back to 1-based index)
    print(current_i + 1, current_j + 1)

if __name__ == "__main__":
    main()