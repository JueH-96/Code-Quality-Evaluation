def main():
    import sys
    from collections import deque
    
    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    S = input_data[2:]
    
    # The player starts at (2,2) in 1-based indexing, i.e. (1,1) in 0-based.
    start_r, start_c = 1, 1
    
    # Precompute for each cell (r,c) the final row/col if we slide
    # UP, DOWN, LEFT, RIGHT from that cell.
    # We'll store these in 2D arrays:
    #   upStop[r][c], downStop[r][c], leftStop[r][c], rightStop[r][c]
    # representing the row/col we finally rest on if we slide in that direction.
    
    upStop   = [[0]*M for _ in range(N)]
    downStop = [[0]*M for _ in range(N)]
    leftStop = [[0]*M for _ in range(N)]
    rightStop= [[0]*M for _ in range(N)]
    
    # Initialize them so that by default we don't move from a rock cell.
    # (Though we won't ever queue rock cells in BFS anyway.)
    for r in range(N):
        for c in range(M):
            upStop[r][c] = r
            downStop[r][c] = r
            leftStop[r][c] = c
            rightStop[r][c] = c
    
    # Build upStop and downStop column by column.
    for c in range(M):
        # Identify all rows that are rocks in ascending order.
        rock_rows = []
        for r in range(N):
            if S[r][c] == '#':
                rock_rows.append(r)
        
        # For the segment between rock_rows[i] and rock_rows[i+1],
        # all '.' cells in that vertical strip slide UP to rock_rows[i]+1
        # and slide DOWN to rock_rows[i+1]-1.
        for i in range(len(rock_rows)-1):
            top_rock = rock_rows[i]
            bottom_rock = rock_rows[i+1]
            # fill for the range (top_rock+1 .. bottom_rock-1)
            for row in range(top_rock+1, bottom_rock):
                if S[row][c] == '.':
                    upStop[row][c] = top_rock + 1
                    downStop[row][c] = bottom_rock - 1
    
    # Build leftStop and rightStop row by row.
    for r in range(N):
        # Identify all columns that are rocks in ascending order.
        rock_cols = []
        for c in range(M):
            if S[r][c] == '#':
                rock_cols.append(c)
        
        for i in range(len(rock_cols)-1):
            left_rock = rock_cols[i]
            right_rock= rock_cols[i+1]
            # fill for the range (left_rock+1 .. right_rock-1)
            for col in range(left_rock+1, right_rock):
                if S[r][col] == '.':
                    leftStop[r][col] = left_rock + 1
                    rightStop[r][col] = right_rock - 1
    
    # We now perform a BFS where each state is a "resting" position (r, c).
    # We also keep track of all squares that get "touched" (reachable)
    # either by resting or by passing over them.
    
    # visitedRest[r][c] tells if we've visited that square as a BFS state.
    visitedRest = [[False]*M for _ in range(N)]
    # reachable[r][c] tells if we've ever touched that square.
    reachable = [[False]*M for _ in range(N)]
    
    # The starting square is (1,1) in 0-based; it's guaranteed to be ice.
    visitedRest[start_r][start_c] = True
    reachable[start_r][start_c] = True
    
    queue = deque()
    queue.append((start_r, start_c))
    
    while queue:
        r, c = queue.popleft()
        
        # Slide UP
        r_up = upStop[r][c]
        if r_up != r:
            # Mark all squares in [min(r, r_up) .. max(r, r_up)] in the same col c as reachable
            if r_up < r:
                for row in range(r_up, r+1):
                    reachable[row][c] = True
            else:
                for row in range(r, r_up+1):
                    reachable[row][c] = True
            # Enqueue the new resting position if not visited
            if not visitedRest[r_up][c] and S[r_up][c] == '.':
                visitedRest[r_up][c] = True
                queue.append((r_up, c))
        
        # Slide DOWN
        r_down = downStop[r][c]
        if r_down != r:
            if r_down < r:
                for row in range(r_down, r+1):
                    reachable[row][c] = True
            else:
                for row in range(r, r_down+1):
                    reachable[row][c] = True
            if not visitedRest[r_down][c] and S[r_down][c] == '.':
                visitedRest[r_down][c] = True
                queue.append((r_down, c))
        
        # Slide LEFT
        c_left = leftStop[r][c]
        if c_left != c:
            if c_left < c:
                for col in range(c_left, c+1):
                    reachable[r][col] = True
            else:
                for col in range(c, c_left+1):
                    reachable[r][col] = True
            if not visitedRest[r][c_left] and S[r][c_left] == '.':
                visitedRest[r][c_left] = True
                queue.append((r, c_left))
        
        # Slide RIGHT
        c_right = rightStop[r][c]
        if c_right != c:
            if c_right < c:
                for col in range(c_right, c+1):
                    reachable[r][col] = True
            else:
                for col in range(c, c_right+1):
                    reachable[r][col] = True
            if not visitedRest[r][c_right] and S[r][c_right] == '.':
                visitedRest[r][c_right] = True
                queue.append((r, c_right))
    
    # Count how many squares are reachable
    answer = 0
    for r in range(N):
        for c in range(M):
            if reachable[r][c]:
                answer += 1
    
    print(answer)

# Do not forget to call main()
if __name__ == "__main__":
    main()