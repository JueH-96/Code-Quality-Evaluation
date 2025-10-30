# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    from collections import deque

    H, W = map(int, sys.stdin.readline().split())
    grid = []
    for _ in range(H):
        row = sys.stdin.readline().strip()
        grid.append(list(row))

    N = int(sys.stdin.readline())
    medicines = {}
    for _ in range(N):
        R_i, C_i, E_i = map(int, sys.stdin.readline().split())
        R_i -=1
        C_i -=1
        medicines[(R_i, C_i)] = E_i

    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                sx, sy = i, j
            if grid[i][j] == 'T':
                tx, ty = i, j

    visited = [[-1]*W for _ in range(H)]
    from collections import deque

    queue = deque()

    # Check if there is medicine at the starting position
    if (sx, sy) in medicines:
        initial_energy = medicines[(sx, sy)]
    else:
        # Cannot move from starting position
        print('No')
        return

    queue.append((sx, sy, initial_energy))
    visited[sx][sy] = initial_energy

    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]

    while queue:
        x, y, energy = queue.popleft()

        if (x, y) == (tx, ty):
            print('Yes')
            return

        # If there is a medicine at current position, and using it increases energy
        if (x, y) in medicines:
            new_energy = medicines[(x, y)]
            if new_energy > energy:
                if visited[x][y] < new_energy:
                    visited[x][y] = new_energy
                    queue.append((x, y, new_energy))
                # Continue to next iteration to avoid double processing
                continue

        for dir in range(4):
            nx = x + dx[dir]
            ny = y + dy[dir]
            if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#':
                if energy > 0:
                    new_energy = energy - 1
                    if visited[nx][ny] < new_energy:
                        visited[nx][ny] = new_energy
                        queue.append((nx, ny, new_energy))

    print('No')

threading.Thread(target=main).start()