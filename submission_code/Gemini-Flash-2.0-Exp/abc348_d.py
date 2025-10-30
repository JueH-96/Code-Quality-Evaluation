from collections import deque

def solve():
    H, W = map(int, input().split())
    grid = [input() for _ in range(H)]
    N = int(input())
    medicines = []
    for _ in range(N):
        r, c, e = map(int, input().split())
        medicines.append((r - 1, c - 1, e))

    start_row, start_col = -1, -1
    goal_row, goal_col = -1, -1
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                start_row, start_col = i, j
            elif grid[i][j] == 'T':
                goal_row, goal_col = i, j

    q = deque([(start_row, start_col, 0, set())])  # row, col, energy, used_medicines
    visited = set()

    while q:
        row, col, energy, used_medicines = q.popleft()

        if (row, col) == (goal_row, goal_col):
            print("Yes")
            return

        if (row, col, energy, tuple(sorted(list(used_medicines)))) in visited:
            continue
        visited.add((row, col, energy, tuple(sorted(list(used_medicines)))))

        # Use medicine
        for i in range(N):
            r, c, e = medicines[i]
            if (r, c) == (row, col) and i not in used_medicines:
                new_used_medicines = set(used_medicines)
                new_used_medicines.add(i)
                q.append((row, col, e, new_used_medicines))

        # Move
        if energy > 0:
            # Move up
            if row > 0 and grid[row - 1][col] != '#':
                q.append((row - 1, col, energy - 1, used_medicines))
            # Move down
            if row < H - 1 and grid[row + 1][col] != '#':
                q.append((row + 1, col, energy - 1, used_medicines))
            # Move left
            if col > 0 and grid[row][col - 1] != '#':
                q.append((row, col - 1, energy - 1, used_medicines))
            # Move right
            if col < W - 1 and grid[row][col + 1] != '#':
                q.append((row, col + 1, energy - 1, used_medicines))

    print("No")

solve()