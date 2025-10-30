def main():
    import sys
    from collections import deque

    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    S = input_data[2:]

    # Precompute the "next stop" in each of the four directions for every ice cell.
    # up[r][c] will store the row-index of the final ice cell if you move UP from (r,c).
    # down[r][c], left[r][c], and right[r][c] similarly.

    up = [[0]*M for _ in range(N)]
    down = [[0]*M for _ in range(N)]
    left_ = [[0]*M for _ in range(N)]
    right_ = [[0]*M for _ in range(N)]

    # Build up[] by scanning each column top-to-bottom
    for c in range(M):
        rock_at = 0  # the row index of the last rock seen so far
        for r in range(N):
            if S[r][c] == '#':
                up[r][c] = r
                rock_at = r
            else:
                up[r][c] = rock_at + 1

    # Build down[] by scanning each column bottom-to-top
    for c in range(M):
        rock_at = N - 1
        for r in range(N-1, -1, -1):
            if S[r][c] == '#':
                down[r][c] = r
                rock_at = r
            else:
                down[r][c] = rock_at - 1

    # Build left_[] by scanning each row left-to-right
    for r in range(N):
        rock_at = 0
        for c in range(M):
            if S[r][c] == '#':
                left_[r][c] = c
                rock_at = c
            else:
                left_[r][c] = rock_at + 1

    # Build right_[] by scanning each row right-to-left
    for r in range(N):
        rock_at = M - 1
        for c in range(M-1, -1, -1):
            if S[r][c] == '#':
                right_[r][c] = c
                rock_at = c
            else:
                right_[r][c] = rock_at - 1

    # We'll do a BFS, starting from (1,1) in 0-based indexing (the problem states (2,2) in 1-based).
    visited = [[False]*M for _ in range(N)]
    q = deque()
    start_r, start_c = 1, 1  # 0-based for (2,2)
    visited[start_r][start_c] = True
    q.append((start_r, start_c))

    while q:
        r, c = q.popleft()
        # Move UP
        nr = up[r][c]
        if nr < r:  # there is space to move upwards
            for rr in range(r-1, nr-1, -1):
                if not visited[rr][c]:
                    visited[rr][c] = True
                    q.append((rr, c))
        # Move DOWN
        nr = down[r][c]
        if nr > r:  # there is space to move downwards
            for rr in range(r+1, nr+1):
                if not visited[rr][c]:
                    visited[rr][c] = True
                    q.append((rr, c))
        # Move LEFT
        nc = left_[r][c]
        if nc < c:  # there is space to move left
            for cc in range(c-1, nc-1, -1):
                if not visited[r][cc]:
                    visited[r][cc] = True
                    q.append((r, cc))
        # Move RIGHT
        nc = right_[r][c]
        if nc > c:  # there is space to move right
            for cc in range(c+1, nc+1):
                if not visited[r][cc]:
                    visited[r][cc] = True
                    q.append((r, cc))

    # Count how many ice-squares are visited
    ans = 0
    for r in range(N):
        for c in range(M):
            if visited[r][c] and S[r][c] == '.':
                ans += 1

    print(ans)

# Do not forget to call main()
main()