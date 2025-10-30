def main():
    import sys
    input = sys.stdin.readline

    N, M = map(int, input().split())
    grid = [input().rstrip() for _ in range(N)]

    # Offsets of the 7 cells adjacent to the top-left 3x3 block
    offsets_tl = [
        (0, 3), (1, 3), (2, 3),
        (3, 0), (3, 1), (3, 2), (3, 3)
    ]
    # Offsets of the 7 cells adjacent to the bottom-right 3x3 block
    offsets_br = [
        (5, 5), (5, 6), (5, 7), (5, 8),
        (6, 5), (7, 5), (8, 5)
    ]

    results = []
    # Slide a 9x9 window with top-left corner at (i, j)
    for i in range(N - 8):
        for j in range(M - 8):
            ok = True
            # Check top-left 3x3 block is all '#'
            for di in range(3):
                for dj in range(3):
                    if grid[i + di][j + dj] != '#':
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue

            # Check bottom-right 3x3 block is all '#'
            for di in range(6, 9):
                for dj in range(6, 9):
                    if grid[i + di][j + dj] != '#':
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue

            # Check the 7 cells adjacent to the TL block are all '.'
            for di, dj in offsets_tl:
                if grid[i + di][j + dj] != '.':
                    ok = False
                    break
            if not ok:
                continue

            # Check the 7 cells adjacent to the BR block are all '.'
            for di, dj in offsets_br:
                if grid[i + di][j + dj] != '.':
                    ok = False
                    break
            if not ok:
                continue

            # If all conditions passed, record the 1-based coordinates
            results.append((i + 1, j + 1))

    # Print in lex order (i then j), which is the order we collected them
    for x, y in results:
        print(x, y)

# Call main to execute
main()