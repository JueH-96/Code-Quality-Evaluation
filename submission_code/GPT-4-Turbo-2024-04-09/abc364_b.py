def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    Si = int(data[2]) - 1  # Convert to 0-based index
    Sj = int(data[3]) - 1  # Convert to 0-based index
    
    grid = []
    index = 4
    for _ in range(H):
        grid.append(data[index])
        index += 1
    
    X = data[index]
    
    # Current position of Takahashi
    x, y = Si, Sj
    
    # Process each move in X
    for move in X:
        if move == 'L':
            if y > 0 and grid[x][y-1] == '.':
                y -= 1
        elif move == 'R':
            if y < W - 1 and grid[x][y+1] == '.':
                y += 1
        elif move == 'U':
            if x > 0 and grid[x-1][y] == '.':
                x -= 1
        elif move == 'D':
            if x < H - 1 and grid[x+1][y] == '.':
                x += 1
    
    # Output the final position (convert back to 1-based index)
    print(x + 1, y + 1)

if __name__ == "__main__":
    main()