def solve():
    n, q = map(int, input().split())
    grid = [input() for _ in range(n)]

    def count_blacks(a, b, c, d):
        count = 0
        for i in range(a, c + 1):
            for j in range(b, d + 1):
                if grid[i % n][j % n] == 'B':
                    count += 1
        return count

    for _ in range(q):
        a, b, c, d = map(int, input().split())
        print(count_blacks(a, b, c, d))

solve()