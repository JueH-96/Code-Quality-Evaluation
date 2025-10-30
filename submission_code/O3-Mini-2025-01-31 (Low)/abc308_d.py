def main():
    import sys
    from collections import deque

    input = sys.stdin.readline

    H, W = map(int, input().split())
    grid = [input().strip() for _ in range(H)]
    
    # Pattern to follow: "snuke"
    pattern = "snuke"
    n_pattern = len(pattern)
    
    # Check if starting and ending cells match the pattern's expected letters.
    if grid[0][0] != pattern[0]:
        print("No")
        return
    # For destination: (H-1, W-1) we don't know what step number it should be because path length is free.
    # So we don't do a check.

    # visited[i][j][p] indicates we visited cell (i,j) with index p (next pattern index to match is p).
    visited = [[[False] * n_pattern for _ in range(W)] for _ in range(H)]
    
    queue = deque()
    # starting cell (0, 0) with pattern index 0 which is 's'
    queue.append((0, 0, 0))
    visited[0][0][0] = True
    
    # Movement vectors: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        i, j, p = queue.popleft()
        # check if destination reached
        if i == H - 1 and j == W - 1:
            print("Yes")
            return
        # Next required letter index: (p + 1) mod 5 since we are following sequence order.
        next_index = (p + 1) % n_pattern
        next_letter = pattern[next_index]
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                if grid[ni][nj] == next_letter and not visited[ni][nj][next_index]:
                    visited[ni][nj][next_index] = True
                    queue.append((ni, nj, next_index))
    
    print("No")

if __name__ == '__main__':
    main()