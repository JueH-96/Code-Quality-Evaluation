def solve():
    n = int(input())
    grid = [input() for _ in range(n)]

    count = 0
    o_cells = []
    for r in range(n):
        for c in range(n):
            if grid[r][c] == 'o':
                o_cells.append((r, c))

    num_o_cells = len(o_cells)

    for i in range(num_o_cells):
        for j in range(i + 1, num_o_cells):
            for k in range(j + 1, num_o_cells):
                cell1 = o_cells[i]
                cell2 = o_cells[j]
                cell3 = o_cells[k]

                r1, c1 = cell1
                r2, c2 = cell2
                r3, c3 = cell3

                rows = {r1, r2, r3}
                cols = {c1, c2, c3}

                if len(rows) == 2 and len(cols) == 2:
                    count += 1

    print(count)

solve()