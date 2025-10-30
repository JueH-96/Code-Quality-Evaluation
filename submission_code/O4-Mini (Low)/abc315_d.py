import sys
import threading
from collections import deque

def main():
    import sys

    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    H, W = map(int, input().split())
    grid = [input().rstrip('
') for _ in range(H)]

    # active[i][j] == True means the cookie is still present
    active = [bytearray(b'\x01') * W for _ in range(H)]

    # For each row: count of active and uniform color or None
    row_count = [W] * H
    row_color = [None] * H
    for i in range(H):
        first = grid[i][0]
        same = True
        for j in range(1, W):
            if grid[i][j] != first:
                same = False
                break
        if same:
            row_color[i] = first

    # For each column: count of active and uniform color or None
    col_count = [H] * W
    col_color = [None] * W
    for j in range(W):
        first = grid[0][j]
        same = True
        for i in range(1, H):
            if grid[i][j] != first:
                same = False
                break
        if same:
            col_color[j] = first

    # Queues of rows and columns eligible for removal
    row_q = deque()
    col_q = deque()
    in_row_q = [False]*H
    in_col_q = [False]*W

    # Initialize
    for i in range(H):
        if row_color[i] is not None and row_count[i] >= 2:
            row_q.append(i)
            in_row_q[i] = True
    for j in range(W):
        if col_color[j] is not None and col_count[j] >= 2:
            col_q.append(j)
            in_col_q[j] = True

    # Process removals
    while row_q or col_q:
        # Process rows
        while row_q:
            i = row_q.popleft()
            in_row_q[i] = False
            # still valid?
            if row_color[i] is None or row_count[i] < 2:
                continue
            # remove all active in row i
            for j in range(W):
                if active[i][j]:
                    active[i][j] = 0
                    row_count[i] -= 1
                    col_count[j] -= 1
                    # after decreasing col_count, we don't need to enqueue columns:
                    # columns were enqueued initially if uniform,
                    # and col_color never becomes non-None later.
            # row_count[i] is now 0

        # Process columns
        while col_q:
            j = col_q.popleft()
            in_col_q[j] = False
            if col_color[j] is None or col_count[j] < 2:
                continue
            # remove all active in column j
            for i in range(H):
                if active[i][j]:
                    active[i][j] = 0
                    col_count[j] -= 1
                    row_count[i] -= 1
            # col_count[j] is now 0

        # Note: No new rows/cols become eligible because
        # row_color and col_color never gain a character value after start.
        # So the queues drain exactly once.

    # Count remaining
    ans = 0
    for i in range(H):
        ans += sum(active[i])
    print(ans)


if __name__ == "__main__":
    main()