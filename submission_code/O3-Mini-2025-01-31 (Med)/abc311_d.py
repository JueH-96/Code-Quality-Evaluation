def main():
    import sys, collections, bisect
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    grid = [next(it).rstrip() for _ in range(N)]
    # Precompute for each row the positions that are rock.
    row_rocks = []
    for i in range(N):
        rlist = []
        for j in range(M):
            if grid[i][j] == '#':
                rlist.append(j)
        row_rocks.append(rlist)
    # Precompute for each column the positions that are rock.
    col_rocks = []
    for j in range(M):
        clist = []
        for i in range(N):
            if grid[i][j] == '#':
                clist.append(i)
        col_rocks.append(clist)
        
    # visited[r][c] : whether the player has stopped here (a "rest" position)
    visited = [[False]*M for _ in range(N)]
    # touched[r][c] : whether the player ever touched this ice square (either passed on or stopped on it)
    touched = [[False]*M for _ in range(N)]
    
    from collections import deque
    dq = deque()
    # Start at (2,2) in 1-indexed grid => (1,1) in 0-indexed.
    start_r, start_c = 1, 1
    visited[start_r][start_c] = True
    touched[start_r][start_c] = True
    dq.append((start_r, start_c))
    
    # The sliding motion: from a resting cell, pick a direction, and slide until the next cell in that
    # direction is rock. All ice cells along the sliding path are "touched".
    while dq:
        r, c = dq.popleft()
        # Slide right.
        if c+1 < M and grid[r][c+1] == '.':
            # In row r, find next rock to the right.
            pos = bisect.bisect_right(row_rocks[r], c)
            rock_index = row_rocks[r][pos]  # Guaranteed to exist because periphery is rock.
            dest_c = rock_index - 1   # the last ice cell before the rock.
            if dest_c != c:  # if we actually move
                for nc in range(c+1, dest_c+1):
                    if not touched[r][nc]:
                        touched[r][nc] = True
                if not visited[r][dest_c]:
                    visited[r][dest_c] = True
                    dq.append((r, dest_c))
                    
        # Slide left.
        if c-1 >= 0 and grid[r][c-1] == '.':
            pos = bisect.bisect_left(row_rocks[r], c) - 1
            rock_index = row_rocks[r][pos]
            dest_c = rock_index + 1
            if dest_c != c:
                for nc in range(dest_c, c):
                    if not touched[r][nc]:
                        touched[r][nc] = True
                if not visited[r][dest_c]:
                    visited[r][dest_c] = True
                    dq.append((r, dest_c))
                    
        # Slide down.
        if r+1 < N and grid[r+1][c] == '.':
            pos = bisect.bisect_right(col_rocks[c], r)
            rock_index = col_rocks[c][pos]
            dest_r = rock_index - 1
            if dest_r != r:
                for nr in range(r+1, dest_r+1):
                    if not touched[nr][c]:
                        touched[nr][c] = True
                if not visited[dest_r][c]:
                    visited[dest_r][c] = True
                    dq.append((dest_r, c))
                    
        # Slide up.
        if r-1 >= 0 and grid[r-1][c] == '.':
            pos = bisect.bisect_left(col_rocks[c], r) - 1
            rock_index = col_rocks[c][pos]
            dest_r = rock_index + 1
            if dest_r != r:
                for nr in range(dest_r, r):
                    if not touched[nr][c]:
                        touched[nr][c] = True
                if not visited[dest_r][c]:
                    visited[dest_r][c] = True
                    dq.append((dest_r, c))
    
    # Count the number of ice squares that the player has touched.
    ans = 0
    for i in range(N):
        for j in range(M):
            if touched[i][j] and grid[i][j] == '.':
                ans += 1
    sys.stdout.write(str(ans))
    
if __name__ == "__main__":
    main()