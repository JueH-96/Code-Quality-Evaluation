from collections import deque

def main():
    H, W = map(int, input().split())
    grid = [input() for _ in range(H)]
    
    # snuke pattern
    pattern = "snuke"
    
    # directions: right, down, left, up
    dx = [0, 1, 0, -1]
    dy = [1, 0, -1, 0]
    
    # BFS state: (x, y, pattern_pos)
    visited = set()
    queue = deque([(0, 0, 0)])  # start at (0,0) with pattern pos 0
    visited.add((0, 0, 0))
    
    while queue:
        x, y, pos = queue.popleft()
        
        # if reached destination and matches pattern
        if x == H-1 and y == W-1:
            if grid[x][y] == pattern[pos]:
                print("Yes")
                return
            continue
            
        # try all 4 directions
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]
            npos = (pos + 1) % 5
            
            # check if new position is valid
            if 0 <= nx < H and 0 <= ny < W:
                # check if next cell matches required pattern
                if grid[nx][ny] == pattern[npos]:
                    state = (nx, ny, npos)
                    if state not in visited:
                        visited.add(state)
                        queue.append(state)
    
    print("No")

if __name__ == "__main__":
    main()