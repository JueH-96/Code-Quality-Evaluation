def main() -> None:
    import sys

    N = int(sys.stdin.readline())
    C = (N + 1) // 2            # centre coordinate, 1-based

    # build the spiral order -------------------------------------
    path = []
    for layer in range(N // 2):         # number of full rectangles
        r0, r1 = 1 + layer, N - layer   # rows   of this rectangle
        c0, c1 = 1 + layer, N - layer   # columns of this rectangle

        # top side
        for c in range(c0, c1 + 1):
            path.append((r0, c))
        # right side
        for r in range(r0 + 1, r1 + 1):
            path.append((r, c1))
        # bottom side
        for c in range(c1 - 1, c0 - 1, -1):
            path.append((r1, c))
        # left side
        for r in range(r1 - 1, r0, -1):
            path.append((r, c0))

    # remove centre if it slipped in (only happens when N == 3)
    if path and path[-1] == (C, C):
        path.pop()

    # -------------------------------------------------------------
    board = [['' for _ in range(N)] for _ in range(N)]
    for num, (r, c) in enumerate(path, 1):
        board[r - 1][c - 1] = str(num)
    board[C - 1][C - 1] = 'T'

    out_lines = [' '.join(row) for row in board]
    sys.stdout.write('
'.join(out_lines))


if __name__ == '__main__':
    main()