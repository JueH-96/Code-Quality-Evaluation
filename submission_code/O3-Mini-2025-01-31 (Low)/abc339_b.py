def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    H = int(input_data[0])
    W = int(input_data[1])
    N = int(input_data[2])
    
    # create grid: False for white, True for black
    grid = [[False] * W for _ in range(H)]
    
    # directions: 0: up, 1: right, 2: down, 3: left
    # initial position: (0, 0) (1,1 in 1-indexed) and direction = up (0)
    pos_r, pos_c = 0, 0
    direction = 0
    # mapping dx, dy changes for each direction index: up, right, down, left.
    # up: (-1, 0), right: (0, 1), down: (1, 0), left: (0, -1)
    dir_moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    for _ in range(N):
        # Check the current cell's color.
        if not grid[pos_r][pos_c]:  # white cell
            # Paint it black
            grid[pos_r][pos_c] = True
            # rotate clockwise: increment direction index
            direction = (direction + 1) % 4
        else:   # black cell
            # Paint it white
            grid[pos_r][pos_c] = False
            # rotate counterclockwise: decrement direction index
            direction = (direction - 1) % 4
        
        # Move forward one cell in the current direction, wrapping around if needed.
        dr, dc = dir_moves[direction]
        pos_r = (pos_r + dr) % H
        pos_c = (pos_c + dc) % W
    
    # Print the grid, black cells '#' and white cells '.'
    output_lines = []
    for row in grid:
        # row: True = black ('#'), False = white ('.')
        line = ''.join('#' if cell else '.' for cell in row)
        output_lines.append(line)
    sys.stdout.write('
'.join(output_lines))
    
if __name__ == '__main__':
    main()