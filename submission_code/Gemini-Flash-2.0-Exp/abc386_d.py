def solve():
    n, m = map(int, input().split())
    constraints = []
    for _ in range(m):
        x, y, c = input().split()
        constraints.append((int(x), int(y), c))

    def check(row_cutoff, col_cutoff):
        for x, y, c in constraints:
            if (x <= row_cutoff and y <= col_cutoff and c == 'W'):
                return False
            if (x > row_cutoff and y > col_cutoff and c == 'B'):
                return False
            if (x <= row_cutoff and y > col_cutoff and c == 'B'):
                return False
            if (x > row_cutoff and y <= col_cutoff and c == 'W'):
                return False
        return True

    for row_cutoff in range(n + 1):
        for col_cutoff in range(n + 1):
            if check(row_cutoff, col_cutoff):
                print("Yes")
                return
    print("No")

solve()