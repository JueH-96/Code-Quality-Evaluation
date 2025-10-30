def solve():
    n = int(input())
    x = list(map(int, input().split()))

    if n == 4:
        m = (x[0] + x[3]) / 2
        x[1] = m + (m - x[1])
        x[2] = m - (x[2] - m)
        print(sum(x))
    else:
        print(sum(x))

solve()