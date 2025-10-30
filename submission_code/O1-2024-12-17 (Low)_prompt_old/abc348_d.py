def solve():
    import sys
    from collections import deque

    input_data = sys.stdin.read().strip().split()
    # Parse inputs
    H, W = map(int, input_data[:2])
    A = input_data[2:2+H]
    idx = 2 + H
    N = int(input_data[idx]); idx += 1

    # Locate S and T; store board as a list of strings or a 2D array
    start = None
    target = None
    board = []
    for i in range(H):
        row_str = A[i]
        row_list = []
        for j, ch in enumerate(row_str):
            if ch == 'S':
                start = (i, j)
                row_list.append('.')  # treat 'S' as empty for movement
            elif ch == 'T':
                target = (i, j)
                row_list.append('.')  # treat 'T' as empty for movement
            else:
                row_list.append(ch)
        board.append(row_list)

    # Read medicine info
    # We'll store at most one medicine per cell (as guaranteed by the problem).
    # med[(r, c)] = E value if there's medicine, else no entry
    med = {}
    for _ in range(N):
        R_i = int(input_data[idx]) - 1; idx += 1
        C_i = int(input_data[idx]) - 1; idx += 1
        E_i = int(input_data[idx]);     idx += 1
        med[(R_i, C_i)] = E_i

    # If for any reason we didn't find S or T, answer No (should not happen by constraints)
    if not start or not target:
        print("No")
        return

    # dist[r][c] will hold the maximum energy we can have upon arriving at (r, c).
    # Initialize with -1 (unreachable).
    dist = [[-1]*W for _ in range(H)]
    sr, sc = start
    tr, tc = target

    # We start at S with 0 energy
    dist[sr][sc] = 0

    # We'll use a queue (BFS-like). Store cells (r,c) where a potential update happened.
    # We will repeatedly process them until no further improvement.
    queue = deque()
    queue.append((sr, sc))

    # Directions for up/down/left/right
    directions = [(-1,0),(1,0),(0,-1),(0,1)]

    # BFS until no more improvements
    while queue:
        r, c = queue.popleft()
        curr_e = dist[r][c]

        # 1) Possibly use medicine in the current cell (if any) and if it improves energy
        if (r, c) in med:
            e_med = med[(r, c)]
            if e_med > curr_e:
                # Update dist[r][c] to e_med
                dist[r][c] = e_med
                # Remove (use) that medicine so it can't be used again
                del med[(r, c)]
                # Put (r,c) back to queue to reprocess neighbors with updated energy
                queue.appendleft((r, c))
                continue  # after picking medicine, re-check from the same spot
                           # (the continue ensures we don't mix old curr_e with new)

        # 2) Move to neighbors if we have enough energy (curr_e > 0)
        if curr_e > 0:
            new_e = curr_e - 1
            for dr, dc in directions:
                nr, nc = r+dr, c+dc
                if 0 <= nr < H and 0 <= nc < W:
                    if board[nr][nc] != '#':  # must be passable
                        if new_e > dist[nr][nc]:
                            dist[nr][nc] = new_e
                            queue.append((nr, nc))

    # Finally, check if we reached T
    if dist[tr][tc] >= 0:
        print("Yes")
    else:
        print("No")