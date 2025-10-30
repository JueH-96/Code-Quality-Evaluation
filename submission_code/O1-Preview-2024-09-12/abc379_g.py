# YOUR CODE HERE
import sys
import threading

def main():
    sys.setrecursionlimit(1 << 25)
    import sys

    MOD = 998244353
    H, W = map(int, sys.stdin.readline().split())
    grid = [list(sys.stdin.readline().strip()) for _ in range(H)]
    n = H * W
    nodes = [(i, j) for i in range(H) for j in range(W)]
    visited = [[False] * W for _ in range(H)]
    res = 1

    dx = [1, 0, -1, 0]
    dy = [0, 1, 0, -1]

    for i in range(H):
        for j in range(W):
            if not visited[i][j]:

                queue = [(i, j)]
                visited[i][j] = True
                pos = []
                color = {}
                flag = True

                color_id = {}
                stack = [(i, j, 0)]
                color[(i, j)] = 0
                while stack:
                    x, y, col = stack.pop()
                    pos.append((x, y))
                    if grid[x][y] in '123':
                        digit = int(grid[x][y]) - 1
                        if digit != col:
                            flag = False
                            break
                    for k in range(4):
                        nx = x + dx[k]
                        ny = y + dy[k]
                        if 0 <= nx < H and 0 <= ny < W:
                            if not visited[nx][ny]:
                                visited[nx][ny] = True
                                stack.append((nx, ny, 1 - col))
                                color[(nx, ny)] = 1 - col
                            else:
                                if color[(nx, ny)] == col:
                                    flag = False
                                    break
                    if not flag:
                        break

                if not flag:
                    res = 0
                    break

                count = [set(), set()]
                for x, y in pos:
                    col = color[(x, y)]
                    if grid[x][y] in '123':
                        digit = int(grid[x][y]) - 1
                        count[col] = {digit}
                    else:
                        count[col].update([0, 1, 2])

                ways = 0
                for c0 in count[0]:
                    for c1 in count[1]:
                        if c0 != c1:
                            ways += 1
                res = res * ways % MOD
    print(res)

threading.Thread(target=main).start()