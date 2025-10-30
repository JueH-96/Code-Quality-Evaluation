import sys

def main():
    N = int(sys.stdin.readline())

    grid = [[0 for _ in range(N)] for _ in range(N)]

    center_coord = N // 2
    grid[center_coord][center_coord] = 'T'

    current_num = 1
    r, c = 0, 0 # Current coordinates, start at top-left (0,0)

    # Loop for each layer k, from outermost (k=0) to N//2 - 1.
    # N//2 gives the number of layers. E.g., N=3, N//2=1 -> k=0. N=5, N//2=2 -> k=0,1.
    for k in range(N // 2):
        # If k > 0, we are starting a new inner layer.
        # The previous layer (k-1) path ended at coordinates (k, k-1).
        # (After the "Move Up" segment of layer k-1, (r,c) = ((k-1)+1, k-1) = (k, k-1) ).
        # We move to (k,k) to start the current layer k.
        if k > 0:
            c += 1 # Move one step right from (k, k-1) to (k, k). r is already k.
        
        # Current (r,c) is the top-left corner of the current layer's square: (k,k).
        
        # Side length of the square subgrid for this spiral layer.
        # For layer k, the square is from (k,k) to (N-1-k, N-1-k).
        # Its side length is (N-1-k) - k + 1 = N - 2k.
        side_len_of_current_square = N - 2 * k

        # Fill the first cell of this layer's path (the top-left corner (k,k))
        grid[r][c] = current_num
        current_num += 1

        # Move Right along the top edge.
        # Number of steps = side_len_of_current_square - 1.
        for _ in range(side_len_of_current_square - 1):
            c += 1
            grid[r][c] = current_num
            current_num += 1
        # (r,c) is now (k, N-1-k)
        
        # Move Down along the right edge.
        # Number of steps = side_len_of_current_square - 1.
        for _ in range(side_len_of_current_square - 1):
            r += 1
            grid[r][c] = current_num
            current_num += 1
        # (r,c) is now (N-1-k, N-1-k)
            
        # Move Left along the bottom edge.
        # Number of steps = side_len_of_current_square - 1.
        for _ in range(side_len_of_current_square - 1):
            c -= 1
            grid[r][c] = current_num
            current_num += 1
        # (r,c) is now (N-1-k, k)
            
        # Move Up along the left edge.
        # Number of steps = side_len_of_current_square - 2.
        # This segment is shorter to avoid overwriting (k,k) and to end at (k+1, k).
        # This positions (r,c) for the transition to layer (k+1),
        # which starts at (k+1, k+1) after the `c += 1` step.
        for _ in range(side_len_of_current_square - 2):
            r -= 1
            grid[r][c] = current_num
            current_num += 1
        # (r,c) is now (k+1, k)
            
    for i in range(N):
        # Print each row. `print(*row)` unpacks `row` into arguments.
        # `print` then converts each argument to string and prints them space-separated.
        print(*(grid[i]))

if __name__ == '__main__':
    main()