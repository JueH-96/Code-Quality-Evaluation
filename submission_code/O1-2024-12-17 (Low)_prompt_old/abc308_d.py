def solve():
    import sys
    from collections import deque

    input_data = sys.stdin.read().strip().split()
    H, W = map(int, input_data[:2])
    grid = input_data[2:]

    # The repeating pattern
    pattern = "snuke"

    # Quick check: if starting cell doesn't match 's', no need to proceed
    if grid[0][0] != 's':
        print("No")
        return

    # Directions for adjacent cells
    directions = [(1,0),(-1,0),(0,1),(0,-1)]

    # visited[r][c][p] = True means we've visited cell (r, c) with pattern index p
    visited = [[[False]*5 for _ in range(W)] for __ in range(H)]

    # BFS queue will hold tuples of (row, col, pattern_index)
    queue = deque()
    visited[0][0][0] = True
    queue.append((0, 0, 0))

    while queue:
        r, c, p_idx = queue.popleft()
        # If we reached (H-1, W-1), we have a valid path
        if (r == H-1) and (c == W-1):
            print("Yes")
            return

        # Move to the next expected letter
        next_idx = (p_idx + 1) % 5
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                # Check if the neighbor's letter matches the pattern
                if not visited[nr][nc][next_idx] and grid[nr][nc] == pattern[next_idx]:
                    visited[nr][nc][next_idx] = True
                    queue.append((nr, nc, next_idx))

    # If BFS completes without reaching (H-1, W-1), no valid path exists
    print("No")

def main():
    solve()

if __name__ == "__main__":
    main()