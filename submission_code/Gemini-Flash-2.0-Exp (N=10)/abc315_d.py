def solve():
    h, w = map(int, input().split())
    grid = [list(input()) for _ in range(h)]

    while True:
        marked = set()
        
        # Mark rows
        for r in range(h):
            row = grid[r]
            if len(row) >= 2:
                first_char = None
                all_same = True
                for c in row:
                    if c != '.':
                        if first_char is None:
                            first_char = c
                        elif c != first_char:
                            all_same = False
                            break
                if all_same and first_char is not None:
                    for col in range(w):
                        if grid[r][col] != '.':
                            marked.add((r, col))

        # Mark columns
        for c in range(w):
            col = [grid[r][c] for r in range(h)]
            if len(col) >= 2:
                first_char = None
                all_same = True
                for r_idx, char in enumerate(col):
                    if char != '.':
                        if first_char is None:
                            first_char = char
                        elif char != first_char:
                            all_same = False
                            break
                if all_same and first_char is not None:
                    for r in range(h):
                        if grid[r][c] != '.':
                            marked.add((r, c))

        if not marked:
            break

        for r, c in marked:
            grid[r][c] = '.'

    count = 0
    for r in range(h):
        for c in range(w):
            if grid[r][c] != '.':
                count += 1
    print(count)

solve()