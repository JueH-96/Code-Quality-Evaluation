def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    N = int(data[2])
    T = data[3]
    grid = data[4:4+H]
    
    # Directions corresponding to 'L', 'R', 'U', 'D'
    direction_map = {
        'L': (0, -1),
        'R': (0, 1),
        'U': (-1, 0),
        'D': (1, 0)
    }
    
    # Calculate the net movement in rows and columns
    net_row_move = 0
    net_col_move = 0
    
    for move in T:
        dr, dc = direction_map[move]
        net_row_move += dr
        net_col_move += dc
    
    possible_positions = set()
    
    # Check all possible starting points in the grid
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                # Calculate the ending position if starting from (i, j)
                end_row = i + net_row_move
                end_col = j + net_col_move
                
                # Check if the end position is valid
                if 0 <= end_row < H and 0 <= end_col < W and grid[end_row][end_col] == '.':
                    possible_positions.add((end_row, end_col))
    
    print(len(possible_positions))

if __name__ == "__main__":
    main()