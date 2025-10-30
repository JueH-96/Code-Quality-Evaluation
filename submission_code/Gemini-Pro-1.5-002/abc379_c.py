def solve():
    n, m = map(int, input().split())
    x = list(map(int, input().split()))
    a = list(map(int, input().split()))

    if sum(a) != n:
        print(-1)
        return

    cells = [0] * (n + 1)
    for i in range(m):
        cells[x[i]] = a[i]

    ans = 0
    for i in range(1, n + 1):
        if cells[i] > 1:
            ans += cells[i] - 1
            cells[i+1] += cells[i] -1
            cells[i] = 1
        elif cells[i] == 0:
            if cells[i-1] > 1:
                ans += 1
                cells[i-1] -= 1
                cells[i] = 1
            else:
                print(-1)
                return
    print(ans)
solve()