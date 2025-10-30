import sys

def main() -> None:
    sys.setrecursionlimit(10000000)
    H, W, K = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]

    # Flatten (i, j) to single index: idx = i * W + j
    def idx(i: int, j: int) -> int:
        return i * W + j

    empty_cells = []
    neighbours = [[] for _ in range(H * W)]

    # Build list of empty cells and their neighbours
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                cur = idx(i, j)
                empty_cells.append(cur)
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == '.':
                        neighbours[cur].append(idx(ni, nj))

    # If the path would need more distinct squares than available, impossible
    if K + 1 > len(empty_cells):
        print(0)
        return

    visited = [False] * (H * W)
    total_paths = 0

    def dfs(current: int, remaining: int) -> None:
        """
        current : current position (flattened index)
        remaining : how many moves still have to be made
        """
        nonlocal total_paths
        if remaining == 0:
            total_paths += 1
            return
        for nxt in neighbours[current]:
            if not visited[nxt]:
                visited[nxt] = True
                dfs(nxt, remaining - 1)
                visited[nxt] = False

    # Start DFS from every empty cell
    for start in empty_cells:
        visited[start] = True
        dfs(start, K)
        visited[start] = False

    print(total_paths)


if __name__ == "__main__":
    main()