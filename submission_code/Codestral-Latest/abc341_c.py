# YOUR CODE HERE
import sys
from collections import deque

def bfs(grid, start, moves):
    H, W = len(grid), len(grid[0])
    directions = {'L': (0, -1), 'R': (0, 1), 'U': (-1, 0), 'D': (1, 0)}
    queue = deque([start])
    visited = set()
    visited.add(start)

    for move in moves:
        dx, dy = directions[move]
        new_queue = deque()
        while queue:
            x, y = queue.popleft()
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] == '.':
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    new_queue.append((nx, ny))
        queue = new_queue

    return len(visited)

def main():
    input = sys.stdin.read
    data = input().split()

    H, W, N = map(int, data[:3])
    T = data[3]
    grid = data[4:4 + H]

    possible_starts = []

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            if grid[i][j] == '.':
                possible_starts.append((i, j))

    result = 0
    for start in possible_starts:
        if bfs(grid, start, T) == 1:
            result += 1

    print(result)

if __name__ == "__main__":
    main()