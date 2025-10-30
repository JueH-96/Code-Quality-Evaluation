def solve():
    n = int(input())
    p = list(map(int, input().split()))

    if n == 1:
        print(0)
        return

    p1 = p[0]
    max_other = 0
    for i in range(1, n):
        max_other = max(max_other, p[i])

    if p1 > max_other:
        print(0)
    else:
        print(max_other - p1 + 1)

solve()