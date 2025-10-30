import collections
import sys

def solve():
    # Read H and W
    H, W = map(int, sys.stdin.readline().split())
    
    # Read the grid
    # S_grid[r][c] will be the character at 0-indexed row r, 0-indexed column c
    S_grid = [sys.stdin.readline().strip() for _ in range(H)]

    # The target character pattern
    target_pattern = "snuke"
    pattern_len = len(target_pattern)

    # Initial check: The starting cell (0,0) must be 's' (target_pattern[0]).
    # The first cell in the path is (0,0), which corresponds to t=1 in the problem's 1-indexed path numbering.
    # The character must be target_pattern[(1-1)%pattern_len] = target_pattern[0].
    if S_grid[0][0] != target_pattern[0]:
        print("No")
        return

    # visited[r][c][k_idx] is True if cell (r,c) can be reached,
    # such that S_grid[r][c] is the character target_pattern[k_idx],
    # and k_idx represents (path_length_to_rc - 1) % pattern_len.
    # Initialize all to False.
    visited = [[[False for _ in range(pattern_len)] for _ in range(W)] for _ in range(H)]

    # Queue for BFS. Stores tuples: (row, col, k_idx_for_this_cell)
    # k_idx_for_this_cell is the index in target_pattern that S_grid[row][col] matches.
    q = collections.deque()

    # Start BFS from (0,0). It's the 1st cell in path (path_length=1).
    # So, its k_idx = (1-1) % pattern_len = 0.
    # We already checked S_grid[0][0] == target_pattern[0] above.
    q.append((0, 0, 0)) # (row, col, k_idx for S_grid[row][col])
    visited[0][0][0] = True

    # Movement deltas for neighbors: up, down, left, right (or any standard 4 directions)
    dr = [-1, 1, 0, 0]
    dc = [0, 0, -1, 1]

    # BFS loop
    while q:
        r, c, current_k_idx = q.popleft()

        # Determine the k_idx for the next cell in the path.
        # If (r,c) was the t-th cell in the path (1-indexed), then current_k_idx = (t-1)%pattern_len.
        # A neighbor (nr,nc) would be the (t+1)-th cell.
        # Its k_idx will be ((t+1)-1)%pattern_len = t%pattern_len.
        # This is equivalent to (current_k_idx + 1) % pattern_len.
        next_k_idx = (current_k_idx + 1) % pattern_len
        expected_char_at_neighbor = target_pattern[next_k_idx]

        # Explore neighbors
        for i in range(4):
            nr, nc = r + dr[i], c + dc[i]

            # Check if neighbor is within grid boundaries
            if 0 <= nr < H and 0 <= nc < W:
                # Check if neighbor has the correct character for the sequence
                if S_grid[nr][nc] == expected_char_at_neighbor:
                    # If this state (nr, nc, next_k_idx) hasn't been visited yet
                    if not visited[nr][nc][next_k_idx]:
                        visited[nr][nc][next_k_idx] = True
                        q.append((nr, nc, next_k_idx))
    
    # After BFS, check if the destination cell (H-1, W-1) was visited.
    # It could have been visited matching any character in the "snuke" pattern,
    # as long as that character is correct for its position in a valid path.
    # The visited[H-1][W-1][k_idx] = True state already ensures this correctness.
    path_found = False
    for k_idx in range(pattern_len):
        if visited[H-1][W-1][k_idx]:
            path_found = True
            break
            
    if path_found:
        print("Yes")
    else:
        print("No")

# Call the solver function
if __name__ == '__main__':
    solve()