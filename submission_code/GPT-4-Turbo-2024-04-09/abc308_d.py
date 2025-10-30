from collections import deque

def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    grid = [data[i + 2] for i in range(H)]
    
    # The pattern we need to follow
    pattern = "snuke"
    pattern_length = len(pattern)
    
    # Directions for moving in the grid (right, down, left, up)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    # Queue for BFS: stores tuples (row, col, step)
    queue = deque([(0, 0, 0)])
    
    # Visited array to keep track of visited cells with specific step mod pattern_length
    visited = [[[False] * pattern_length for _ in range(W)] for _ in range(H)]
    visited[0][0][0] = True
    
    while queue:
        r, c, step = queue.popleft()
        current_char = grid[r][c]
        expected_char = pattern[step % pattern_length]
        
        # Check if the current cell matches the expected character in the pattern
        if current_char != expected_char:
            continue
        
        # Check if we reached the bottom-right corner with the correct sequence
        if r == H - 1 and c == W - 1 and expected_char == pattern[(H + W - 2) % pattern_length]:
            print("Yes")
            return
        
        # Explore neighbors
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                next_step = step + 1
                next_char_index = next_step % pattern_length
                if not visited[nr][nc][next_char_index]:
                    visited[nr][nc][next_char_index] = True
                    queue.append((nr, nc, next_step))
    
    print("No")