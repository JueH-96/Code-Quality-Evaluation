def main():
    import sys
    from collections import deque
    
    data = sys.stdin.read().split()
    if not data:
        return
    
    H = int(data[0])
    W = int(data[1])
    grid = data[2:2+H]
    
    # The required repeating sequence
    pattern = "snuke"
    
    # The starting cell must have 's'
    if grid[0][0] != pattern[0]:
        sys.stdout.write("No")
        return
    
    # visited[i][j][r] means we have visited cell (i, j) with pattern state r.
    # r corresponds to the index in the string 'snuke' that should equal grid[i][j].
    visited = [[[False] * 5 for _ in range(W)] for _ in range(H)]
    queue = deque()
    queue.append((0, 0, 0))
    visited[0][0][0] = True
    
    # Define four moves: right, down, left, up.
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    while queue:
        i, j, r = queue.popleft()
        # If reached destination, print Yes.
        if i == H - 1 and j == W - 1:
            sys.stdout.write("Yes")
            return
        
        # Next expected letter is the next in the pattern.
        next_r = (r + 1) % 5
        expected_char = pattern[next_r]
        
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                if grid[ni][nj] == expected_char and not visited[ni][nj][next_r]:
                    visited[ni][nj][next_r] = True
                    queue.append((ni, nj, next_r))
    
    sys.stdout.write("No")

if __name__ == '__main__':
    main()