def solve():
    n, t = map(int, input().split())
    a = list(map(int, input().split()))

    marked_cells = set()
    row_counts = [0] * n
    col_counts = [0] * n
    main_diag_count = 0
    anti_diag_count = 0

    for turn, num in enumerate(a, 1):
        if num in marked_cells:
            continue
        marked_cells.add(num)

        row = (num - 1) // n
        col = (num - 1) % n

        row_counts[row] += 1
        col_counts[col] += 1

        if row == col:
            main_diag_count += 1
        if row + col == n - 1:
            anti_diag_count += 1

        if row_counts[row] == n or \
           col_counts[col] == n or \
           main_diag_count == n or \
           anti_diag_count == n:
            print(turn)
            return

    print(-1)

# YOUR CODE HERE
solve()