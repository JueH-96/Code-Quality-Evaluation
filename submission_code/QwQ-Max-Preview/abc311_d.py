import sys
from collections import deque

def main():
    n, m = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(n)]
    
    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    touched = set()
    enqueued = set()
    
    start_x, start_y = 1, 1  # 0-based index for (2,2)
    touched.add((start_x, start_y))
    enqueued.add((start_x, start_y))
    
    queue = deque()
    queue.append((start_x, start_y))
    
    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            path = []
            current_x, current_y = x, y
            while True:
                next_x = current_x + dx
                next_y = current_y + dy
                if next_x < 0 or next_x >= n or next_y < 0 or next_y >= m:
                    break
                if grid[next_x][next_y] == '#':
                    break
                current_x, current_y = next_x, next_y
                path.append((current_x, current_y))
            # Add all squares in the path to touched
            for pos in path:
                if pos not in touched:
                    touched.add(pos)
            # Check if the end position needs to be enqueued
            if path:
                end_x, end_y = path[-1]
                if (end_x, end_y) not in enqueued:
                    enqueued.add((end_x, end_y))
                    queue.append((end_x, end_y))
    
    print(len(touched))

if __name__ == "__main__":
    main()