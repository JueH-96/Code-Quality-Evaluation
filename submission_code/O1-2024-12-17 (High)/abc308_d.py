def main():
    import sys
    from collections import deque

    input = sys.stdin.readline
    H, W = map(int, input().split())
    grid = [input().strip() for _ in range(H)]

    pattern = "snuke"  # The repeating pattern we require

    # The starting cell must match the first character of the pattern, i.e. 's'
    if grid[0][0] != 's':
        print("No")
        return

    # visited[r][c][m] = True indicates we've reached (r, c) with pattern index m
    visited = [[[False]*5 for _ in range(W)] for _ in range(H)]
    visited[0][0][0] = True

    queue = deque()
    queue.append((0, 0, 0))  # row, col, and current pattern index

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while queue:
        r, c, m = queue.popleft()

        # If we've reached the bottom-right cell, we've found a valid path
        if r == H - 1 and c == W - 1:
            print("Yes")
            return

        nm = (m + 1) % 5
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                if not visited[nr][nc][nm] and grid[nr][nc] == pattern[nm]:
                    visited[nr][nc][nm] = True
                    queue.append((nr, nc, nm))

    # If we exhaust all possibilities without reaching (H, W), the answer is No
    print("No")

# Do not forget to call main()
if __name__ == "__main__":
    main()