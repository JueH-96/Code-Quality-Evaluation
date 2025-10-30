def main():
    import sys
    from collections import deque
    input = sys.stdin.readline
    
    H, W = map(int, input().split())
    grid = [input().strip() for _ in range(H)]
    
    # The pattern of letters we need to follow
    pattern = "snuke"
    
    # Check if starting letter is correct
    if grid[0][0] != pattern[0]:
        print("No")
        return
    
    # visited[i][j][k] where k is the index in pattern for that state.
    visited = [[[False]*5 for _ in range(W)] for _ in range(H)]
    dq = deque()
    
    # Start state: top-left cell, pattern index = 0 ('s')
    dq.append((0, 0, 0))
    visited[0][0][0] = True
    
    # BFS directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while dq:
        i, j, idx = dq.popleft()
        # Check if we reached destination cell
        if i == H - 1 and j == W - 1:
            print("Yes")
            return
        
        # Next pattern letter's index
        next_idx = (idx + 1) % 5
        next_char = pattern[next_idx]
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if ni < 0 or ni >= H or nj < 0 or nj >= W:
                continue
            if grid[ni][nj] != next_char:
                continue
            if visited[ni][nj][next_idx]:
                continue
            visited[ni][nj][next_idx] = True
            dq.append((ni, nj, next_idx))
    
    print("No")

if __name__ == '__main__':
    main()