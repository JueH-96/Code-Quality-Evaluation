def main():
    import sys
    sys.setrecursionlimit(10**7)
    
    N = int(sys.stdin.readline().strip())
    
    # We'll fill an NxN board in a spiral order. 
    # For an odd N, the spiral will end at the center cell.
    # We assign parts 1..(N^2-1) to the first N^2 - 1 cells of the spiral
    # and place 'T' in the last (center) cell.
    
    # Prepare a 2D array (strings) to store the result
    board = [[""] * N for _ in range(N)]
    
    # Boundaries for the spiral
    top, bottom = 0, N - 1
    left, right = 0, N - 1
    
    # Current position and direction (0=right,1=down,2=left,3=up)
    r, c = 0, 0
    direction = 0
    
    # We will store all cells in spiral order in "path"
    path = []
    
    for _ in range(N * N):
        path.append((r, c))
        if direction == 0:  # moving right
            if c == right:
                # turn down
                direction = 1
                top += 1
                r += 1
            else:
                c += 1
        elif direction == 1:  # moving down
            if r == bottom:
                # turn left
                direction = 2
                right -= 1
                c -= 1
            else:
                r += 1
        elif direction == 2:  # moving left
            if c == left:
                # turn up
                direction = 3
                bottom -= 1
                r -= 1
            else:
                c -= 1
        else:  # direction == 3, moving up
            if r == top:
                # turn right
                direction = 0
                left += 1
                c += 1
            else:
                r -= 1
    
    # Now "path" contains all cells in spiral order,
    # and for odd N, path[-1] is the center cell.
    
    # Place parts 1..(N^2-1) in the first N^2-1 cells of path,
    # and place 'T' in the last cell.
    for i in range(N * N - 1):
        rr, cc = path[i]
        board[rr][cc] = str(i + 1)
    
    # The last cell gets 'T'
    rr, cc = path[-1]
    board[rr][cc] = "T"
    
    # Print the board
    for row in board:
        print(" ".join(row))

# Do not forget to call main()
main()