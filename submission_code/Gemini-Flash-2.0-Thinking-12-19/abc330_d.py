def solve():
    n = int(input())
    grid = [input() for _ in range(n)]
    count = 0
    for r1 in range(n):
        for c1 in range(n):
            if grid[r1][c1] == 'o':
                row_o_count = 0
                for c2 in range(n):
                    if c2 != c1 and grid[r1][c2] == 'o':
                        row_o_count += 1
                col_o_count = 0
                for r2 in range(n):
                    if r2 != r1 and grid[r2][c1] == 'o':
                        col_o_count += 1
                count += row_o_count * col_o_count
    print(count)

if __name__ == '__main__':
    solve()