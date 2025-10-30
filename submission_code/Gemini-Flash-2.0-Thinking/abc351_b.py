def solve():
    n = int(input())
    grid_a = [input() for _ in range(n)]
    grid_b = [input() for _ in range(n)]

    for i in range(n):
        row_a = grid_a[i]
        row_b = grid_b[i]
        if row_a != row_b:
            for j in range(n):
                if row_a[j] != row_b[j]:
                    print(i + 1, j + 1)
                    return

solve()