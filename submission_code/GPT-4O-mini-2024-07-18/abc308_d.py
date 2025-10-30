from collections import deque

def can_reach_path(H, W, grid):
    target = "snukes"
    target_length = len(target)
    
    # Directions for moving in the grid (right, down, left, up)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    # BFS initialization
    queue = deque([(0, 0, 0)])  # (row, col, index in target)
    visited = set((0, 0, 0))  # (row, col, index in target)
    
    while queue:
        x, y, idx = queue.popleft()
        
        # If we reached the bottom-right corner and the index matches the target
        if x == H - 1 and y == W - 1 and idx == 5:
            return "Yes"
        
        # Next character to match
        next_idx = (idx + 1) % target_length
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            if 0 <= nx < H and 0 <= ny < W:
                if grid[nx][ny] == target[next_idx]:
                    if (nx, ny, next_idx) not in visited:
                        visited.add((nx, ny, next_idx))
                        queue.append((nx, ny, next_idx))
    
    return "No"

def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    H, W = map(int, data[0].split())
    grid = [data[i + 1] for i in range(H)]
    
    result = can_reach_path(H, W, grid)
    print(result)

if __name__ == "__main__":
    main()