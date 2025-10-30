# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1000000)
    H, W, K = map(int, sys.stdin.readline().split())
    grid = []
    empty_cells = []
    for i in range(H):
        line = sys.stdin.readline().strip()
        grid.append(line)
        for j in range(W):
            if line[j] == '.':
                empty_cells.append((i, j))

    total_paths = [0]  # use list to make it mutable in nested function

    visited_global = set()
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def dfs(i, j, depth, visited):
        if depth == K:
            total_paths[0] += 1
            return
        for dx, dy in moves:
            ni, nj = i + dx, j + dy
            if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == '.':
                if (ni, nj) not in visited:
                    visited.add((ni, nj))
                    dfs(ni, nj, depth + 1, visited)
                    visited.remove((ni, nj))

    for (i, j) in empty_cells:
        visited = set()
        visited.add((i, j))
        dfs(i, j, 0, visited)

    print(total_paths[0])

threading.Thread(target=main).start()