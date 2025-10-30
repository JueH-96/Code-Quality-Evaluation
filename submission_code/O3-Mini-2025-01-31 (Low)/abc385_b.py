def main():
    import sys
    input_data = sys.stdin.read().splitlines()
    # First line containing H, W, X, Y
    H, W, X, Y = map(int, input_data[0].split())
    
    # Adjust initial indices to be 0-indexed.
    start_r, start_c = X - 1, Y - 1
    
    # Read the grid
    grid = []
    for i in range(1, H+1):
        grid.append(list(input_data[i].rstrip()))
    
    # The move string is in line H+1
    T = input_data[H+1].strip()
    
    # Current position
    r, c = start_r, start_c
    visited_houses = set()
    
    # Movement map dictionary: direction letter -> (dr, dc)
    moves = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
    
    # Process each move in T
    for ch in T:
        dr, dc = moves[ch]
        new_r, new_c = r + dr, c + dc
        # Check if position is within grid bounds and passable (either '.' or '@')
        if 0 <= new_r < H and 0 <= new_c < W and grid[new_r][new_c] != '#':
            r, c = new_r, new_c
        # If the new cell is a house, record it.
        if grid[r][c] == '@':
            visited_houses.add((r, c))
    
    # After executing all moves, output 1-indexed cell (r, c) and the count of distinct houses visited.
    print(r+1, c+1, len(visited_houses))


if __name__ == '__main__':
    main()