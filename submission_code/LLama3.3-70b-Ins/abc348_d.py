from collections import deque
from sys import stdin

def main():
    H, W = map(int, stdin.readline().split())
    grid = [list(stdin.readline().rstrip()) for _ in range(H)]

    start = None
    goal = None
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                start = (i, j)
            elif grid[i][j] == 'T':
                goal = (i, j)

    N = int(stdin.readline())
    medicines = []
    for _ in range(N):
        r, c, e = map(int, stdin.readline().split())
        medicines.append((r-1, c-1, e))

    queue = deque([(start, 0)])
    visited = {(start, 0)}
    while queue:
        (x, y), energy = queue.popleft()

        if (x, y) == goal:
            print("Yes")
            return

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#':
                new_energy = energy - 1
                if (nx, ny) in [med[0] for med in medicines]:
                    for med in medicines:
                        if med[0] == (nx, ny):
                            new_energy = max(new_energy, med[2] - 1)
                            medicines.remove(med)
                            break
                if new_energy >= 0 and (nx, ny, new_energy) not in visited:
                    queue.append(((nx, ny), new_energy))
                    visited.add((nx, ny, new_energy))

    print("No")

if __name__ == "__main__":
    main()