def solve():
    H, W, N = map(int, input().split())
    T = input().strip()
    
    # Read the grid
    grid = []
    for _ in range(H):
        grid.append(input().strip())
    
    # Define the reverse move for each direction
    reverse_move = {'L': 'R', 'R': 'L', 'U': 'D', 'D': 'U'}
    
    # Calculate the reverse instruction sequence
    reverse_T = ''.join(reverse_move[ch] for ch in reversed(T))
    
    possible_positions = 0
    
    # Check each cell in the grid
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':  # If it's a land cell
                current_i, current_j = i, j
                valid = True
                
                # Trace back the path
                for move in reverse_T:
                    if move == 'L':
                        current_j -= 1
                    elif move == 'R':
                        current_j += 1
                    elif move == 'U':
                        current_i -= 1
                    elif move == 'D':
                        current_i += 1
                    
                    # Check if the cell is within bounds and is land
                    if (0 <= current_i < H and 0 <= current_j < W and 
                        grid[current_i][current_j] == '.'):
                        continue
                    else:
                        valid = False
                        break
                
                if valid:
                    possible_positions += 1
    
    return possible_positions

print(solve())