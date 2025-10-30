import sys

def main():
    H, W = map(int, sys.stdin.readline().split())
    Si, Sj = map(int, sys.stdin.readline().split())
    
    # Convert to 0-indexed
    current_r = Si - 1
    current_c = Sj - 1
    
    grid_map = []
    for _ in range(H):
        grid_map.append(sys.stdin.readline().strip())
        
    X = sys.stdin.readline().strip()
    
    for action in X:
        dr, dc = 0, 0 # dr is change in row, dc is change in column
        
        if action == 'L':
            dc = -1
        elif action == 'R':
            dc = 1
        elif action == 'U':
            dr = -1
        elif action == 'D':
            dr = 1
        
        potential_new_r = current_r + dr
        potential_new_c = current_c + dc
        
        # Check if potential new position is valid (within bounds)
        if 0 <= potential_new_r < H and 0 <= potential_new_c < W:
            # Check if the cell is empty
            if grid_map[potential_new_r][potential_new_c] == '.':
                # If all conditions met, perform the move
                current_r = potential_new_r
                current_c = potential_new_c
        # Else (out of bounds or cell not empty): stay in current_r, current_c (i.e., do nothing)
                
    # Convert back to 1-indexed for printing
    print(current_r + 1, current_c + 1)

if __name__ == '__main__':
    main()