def solve():
    n, t = map(int, input().split())
    a = list(map(int, input().split()))

    grid = [[0] * n for _ in range(n)]
    marked = [[False] * n for _ in range(n)]

    def check_bingo():
        for i in range(n):
            all_marked_row = True
            for j in range(n):
                if not marked[i][j]:
                    all_marked_row = False
                    break
            if all_marked_row:
                return True

        for j in range(n):
            all_marked_col = True
            for i in range(n):
                if not marked[i][j]:
                    all_marked_col = False
                    break
            if all_marked_col:
                return True

        all_marked_diag1 = True
        for i in range(n):
            if not marked[i][i]:
                all_marked_diag1 = False
                break
        if all_marked_diag1:
            return True

        all_marked_diag2 = True
        for i in range(n):
            if not marked[i][n - 1 - i]:
                all_marked_diag2 = False
                break
        if all_marked_diag2:
            return True

        return False

    for turn, num in enumerate(a):
        row = (num - 1) // n
        col = (num - 1) % n
        marked[row][col] = True

        if check_bingo():
            print(turn + 1)
            return

    print(-1)

solve()