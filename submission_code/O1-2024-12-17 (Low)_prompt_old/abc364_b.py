def solve():
    import sys
    
    data = sys.stdin.read().strip().split()
    
    H, W = map(int, data[0:2])
    S_i, S_j = map(int, data[2:4])
    
    # Next H lines contain the grid
    grid = data[4:4+H]
    
    # The next token is the string X
    X = data[4+H]
    
    # Convert to 0-based indexing internally
    cur_i = S_i - 1
    cur_j = S_j - 1
    
    # Directions
    moves = {
        'L': (0, -1),
        'R': (0, 1),
        'U': (-1, 0),
        'D': (1, 0),
    }
    
    for ch in X:
        di, dj = moves[ch]
        new_i = cur_i + di
        new_j = cur_j + dj
        
        # Check boundaries
        if 0 <= new_i < H and 0 <= new_j < W:
            # Check if empty
            if grid[new_i][new_j] == '.':
                cur_i, cur_j = new_i, new_j
    
    # Convert back to 1-based indexing before printing
    print(cur_i + 1, cur_j + 1)

def main():
    solve()

if __name__ == "__main__":
    main()