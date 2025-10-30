# YOUR CODE HERE

import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    H, W = map(int, sys.stdin.readline().split())
    grid_color = [list(sys.stdin.readline().strip()) for _ in range(H)]
    rows = [set() for _ in range(H)]
    columns = [set() for _ in range(W)]
    # Initialize rows and columns
    for i in range(H):
        for j in range(W):
            rows[i].add(j)
            columns[j].add(i)
    total_cookies = H * W
    while True:
        marked_positions = set()
        # Check rows
        for i in range(H):
            if len(rows[i]) >= 2:
                colors = set()
                for j in rows[i]:
                    colors.add(grid_color[i][j])
                    if len(colors) > 1:
                        break
                else:
                    # All colors are the same, mark positions
                    for j in rows[i]:
                        marked_positions.add((i, j))
        # Check columns
        for j in range(W):
            if len(columns[j]) >= 2:
                colors = set()
                for i in columns[j]:
                    colors.add(grid_color[i][j])
                    if len(colors) > 1:
                        break
                else:
                    # All colors are the same, mark positions
                    for i in columns[j]:
                        marked_positions.add((i, j))
        if not marked_positions:
            break
        # Remove marked positions
        for i, j in marked_positions:
            if j in rows[i]:
                rows[i].remove(j)
            if i in columns[j]:
                columns[j].remove(i)
            grid_color[i][j] = None  # Mark as removed
        total_cookies -= len(marked_positions)
    print(total_cookies)

threading.Thread(target=main).start()