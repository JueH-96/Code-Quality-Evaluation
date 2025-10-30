def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    # Read dimensions
    parts = data[0].split()
    N = int(parts[0])
    M = int(parts[1])
    grid = data[1:1+N]
    # In our code we use 0-indexing.
    # By problem statement the outer boundary (row 0, row N-1, col 0, col M-1) is rock.
    # The starting cell is (1,1) in 0-index (i.e. square (2,2) in 1-index).

    # Precompute for each cell in the grid the farthest reachable ice cell
    # in the four cardinal directions (up, down, left, right) if you slide.
    # For an ice cell (r,c), the following holds:
    #   • right_end[r][c] is the largest column index in the contiguous block of '.' from (r, c) going right.
    #   • left_end[r][c] is the smallest col index in that contiguous block going left.
    #   • up_end[r][c] is the smallest row index in that contiguous block going up.
    #   • down_end[r][c] is the largest row index in that contiguous block going down.
    #
    # We compute these for all cells. (For rock cells the value is not needed.)
    N_int, M_int = N, M
    right_end = [[0]*M_int for _ in range(N_int)]
    left_end = [[0]*M_int for _ in range(N_int)]
    up_end = [[0]*M_int for _ in range(N_int)]
    down_end = [[0]*M_int for _ in range(N_int)]
    
    # For right_end, process each row from rightmost column to left.
    for r in range(N_int):
        for c in range(M_int-1, -1, -1):
            if grid[r][c] == '.':
                if c == M_int-1:
                    right_end[r][c] = c
                else:
                    if grid[r][c+1] == '.':
                        right_end[r][c] = right_end[r][c+1]
                    else:
                        right_end[r][c] = c
            else:
                right_end[r][c] = c
    # For left_end, process each row from leftmost to right.
    for r in range(N_int):
        for c in range(M_int):
            if grid[r][c] == '.':
                if c == 0:
                    left_end[r][c] = c
                else:
                    if grid[r][c-1] == '.':
                        left_end[r][c] = left_end[r][c-1]
                    else:
                        left_end[r][c] = c
            else:
                left_end[r][c] = c
    # For up_end, process each column from top down.
    for c in range(M_int):
        for r in range(N_int):
            if grid[r][c] == '.':
                if r == 0:
                    up_end[r][c] = r
                else:
                    if grid[r-1][c] == '.':
                        up_end[r][c] = up_end[r-1][c]
                    else:
                        up_end[r][c] = r
            else:
                up_end[r][c] = r
    # For down_end, process each column from bottom up.
    for c in range(M_int):
        for r in range(N_int-1, -1, -1):
            if grid[r][c] == '.':
                if r == N_int-1:
                    down_end[r][c] = r
                else:
                    if grid[r+1][c] == '.':
                        down_end[r][c] = down_end[r+1][c]
                    else:
                        down_end[r][c] = r
            else:
                down_end[r][c] = r

    # We now simulate the allowed moves.
    # The rules: from a resting cell, choose one direction. Then if the immediate cell
    # in that direction is ice, you “slide” until you are about to hit rock.
    # Note that every ice square you pass through (even if not an endpoint) becomes "touched."
    # We will conduct a BFS over the endpoints (cells where you stop sliding)
    # and while doing so mark each cell passed (or started) as touched.
    from collections import deque
    q = deque()
    visited = [[False]*M_int for _ in range(N_int)]
    touched = [[False]*M_int for _ in range(N_int)]
    
    start_r, start_c = 1, 1  # starting cell (2,2) 1-indexed => (1,1) 0-indexed.
    visited[start_r][start_c] = True
    touched[start_r][start_c] = True
    q.append( (start_r, start_c) )
    
    # The four directions.
    # (dr, dc) pairs: up, down, left, right.
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while q:
        r, c = q.popleft()
        for dr, dc in directions:
            nr = r + dr
            nc = c + dc
            # If the immediate next cell is out-of-bounds or rock, skip.
            if not (0 <= nr < N_int and 0 <= nc < M_int):
                continue
            if grid[nr][nc] != '.':
                continue
            # There is a contiguous block in the chosen direction.
            # We use our precomputed arrays to “jump” to the far end.
            if dr == 0 and dc == 1:
                # Moving right: slide along row r starting at column c+1.
                seg_start = c + 1
                seg_end = right_end[r][nc]  # nc is c+1.
                # Mark all cells in row r from col seg_start to seg_end as touched.
                row_touched = touched[r]
                for col in range(seg_start, seg_end + 1):
                    if not row_touched[col]:
                        row_touched[col] = True
                new_r, new_c = r, seg_end
            elif dr == 0 and dc == -1:
                # Moving left: slide along row r from column c-1 down to left_end[r][c-1].
                seg_end = c - 1
                seg_start = left_end[r][nc]  # nc is c-1.
                row_touched = touched[r]
                for col in range(seg_start, seg_end + 1):
                    if not row_touched[col]:
                        row_touched[col] = True
                new_r, new_c = r, seg_start
            elif dr == 1 and dc == 0:
                # Moving down: slide along column c from row r+1 to down_end[r+1][c].
                seg_start = r + 1
                seg_end = down_end[nr][c]  # nr is r+1.
                for row_i in range(seg_start, seg_end + 1):
                    if not touched[row_i][c]:
                        touched[row_i][c] = True
                new_r, new_c = seg_end, c
            elif dr == -1 and dc == 0:
                # Moving up: slide along column c from row r-1 up to up_end[r-1][c].
                seg_end = r - 1
                seg_start = up_end[nr][c]  # nr is r-1.
                for row_i in range(seg_start, seg_end + 1):
                    if not touched[row_i][c]:
                        touched[row_i][c] = True
                new_r, new_c = seg_start, c
            else:
                continue

            # If this new resting cell (endpoint) has not been visited, add it.
            if not visited[new_r][new_c]:
                visited[new_r][new_c] = True
                q.append((new_r, new_c))
                
    # Count the number of touched cells.
    ans = 0
    for r in range(N_int):
        ans += sum(1 for x in touched[r] if x)
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()