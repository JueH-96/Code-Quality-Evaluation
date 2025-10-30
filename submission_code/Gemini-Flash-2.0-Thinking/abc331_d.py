def solve():
    n, q = map(int, input().split())
    pattern = [input() for _ in range(n)]

    def count_black_in_rect(a, b, c, d, pattern, n):
        count = 0
        for r_pat in range(n):
            for c_pat in range(n):
                if pattern[r_pat][c_pat] == 'B':
                    row_start = a
                    row_end = c
                    col_start = b
                    col_end = d

                    row_matches = max(0, (row_end - r_pat + n) // n - (row_start - r_pat + n - 1) // n)
                    col_matches = max(0, (col_end - c_pat + n) // n - (col_start - c_pat + n - 1) // n)
                    count += row_matches * col_matches
        return count

    for _ in range(q):
        a, b, c, d = map(int, input().split())
        result = count_black_in_rect(a, b, c, d, pattern, n)
        print(result)

solve()