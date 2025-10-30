# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    grid = data[2:H+2]
    
    pattern = "snuke"
    
    from collections import deque
    
    # Directions for moving in the grid: right, down, left, up
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    # BFS setup
    queue = deque([(0, 0)])  # Start from the top-left corner (0-indexed)
    visited = set()
    visited.add((0, 0))
    
    while queue:
        x, y = queue.popleft()
        current_step = (x + y) % 5  # Determine which character in "snuke" we expect
        current_char = pattern[current_step]
        
        if (x, y) == (H-1, W-1):
            print("Yes")
            return
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and (nx, ny) not in visited:
                next_step = (nx + ny) % 5
                next_char = pattern[next_step]
                if grid[nx][ny] == next_char:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
    
    print("No")

if __name__ == "__main__":
    main()