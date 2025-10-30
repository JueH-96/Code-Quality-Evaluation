import sys
from itertools import permutations

def can_transform(A, B, H, W):
    def swap_rows(grid, i):
        grid[i], grid[i+1] = grid[i+1], grid[i]

    def swap_cols(grid, j):
        for row in grid:
            row[j], row[j+1] = row[j+1], row[j]

    def bfs(A, B, H, W):
        from collections import deque

        initial_state = tuple(tuple(row) for row in A)
        target_state = tuple(tuple(row) for row in B)

        queue = deque([(initial_state, 0)])
        visited = set([initial_state])

        while queue:
            current, steps = queue.popleft()

            if current == target_state:
                return steps

            current_grid = [list(row) for row in current]

            for i in range(H-1):
                new_grid = [row[:] for row in current_grid]
                swap_rows(new_grid, i)
                new_state = tuple(tuple(row) for row in new_grid)
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, steps + 1))

            for j in range(W-1):
                new_grid = [row[:] for row in current_grid]
                swap_cols(new_grid, j)
                new_state = tuple(tuple(row) for row in new_grid)
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, steps + 1))

        return -1

    return bfs(A, B, H, W)

def main():
    input = sys.stdin.read
    data = input().split()

    H = int(data[0])
    W = int(data[1])

    A = []
    B = []

    index = 2
    for _ in range(H):
        row = list(map(int, data[index:index + W]))
        A.append(row)
        index += W

    for _ in range(H):
        row = list(map(int, data[index:index + W]))
        B.append(row)
        index += W

    result = can_transform(A, B, H, W)
    print(result)

if __name__ == "__main__":
    main()