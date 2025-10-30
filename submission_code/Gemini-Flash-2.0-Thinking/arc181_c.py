def solve():
    n = int(input())
    p = list(map(int, input().split()))
    q = list(map(int, input().split()))

    pos_p = {val: i for i, val in enumerate(p)}
    pos_q = {val: i for i, val in enumerate(q)}

    grid = [[0] * n for _ in range(n)]

    for r in range(1, n + 1):
        for c in range(1, n + 1):
            if pos_p[r] > pos_q[c]:
                grid[r - 1][c - 1] = 1
            else:
                grid[r - 1][c - 1] = 0

    for row in grid:
        print("".join(map(str, row)))

if __name__ == "__main__":
    solve()