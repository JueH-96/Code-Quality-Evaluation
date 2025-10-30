import numpy as np

def shift(array, direction):
    if direction == 'L':
        # Shift left: shift axis=1 by -1
        shifted = np.pad(array, ((0,0),(0,1)), mode='constant', constant_values=False)[:, :-1]
    elif direction == 'R':
        # Shift right: shift axis=1 by +1
        shifted = np.pad(array, ((0,0),(1,0)), mode='constant', constant_values=False)[:, 1:]
    elif direction == 'U':
        # Shift up: shift axis=0 by -1
        shifted = np.pad(array, ((0,1),(0,0)), mode='constant', constant_values=False)[:-1, :]
    elif direction == 'D':
        # Shift down: shift axis=0 by +1
        shifted = np.pad(array, ((1,0),(0,0)), mode='constant', constant_values=False)[1:, :]
    else:
        shifted = array.copy()
    return shifted

def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    H, W, N = map(int, data[0].split())
    T = data[1]
    grid_lines = data[2:2+H]
    
    # Convert grid to boolean array: True for land ('.'), False for sea ('#')
    grid = np.array([[c == '.' for c in row] for row in grid_lines], dtype=bool)
    
    # Create inverse movement sequence
    inverse_map = {'L':'R', 'R':'L', 'U':'D', 'D':'U'}
    inverse_T = [inverse_map[m] for m in reversed(T)]
    
    # Initialize possible positions as land cells
    possible = grid.copy()
    
    # Apply each inverse movement
    for delta in inverse_T:
        shifted = shift(possible, delta)
        possible = shifted & grid
    
    # Count the number of possible current positions
    print(np.sum(possible))

if __name__ == "__main__":
    main()