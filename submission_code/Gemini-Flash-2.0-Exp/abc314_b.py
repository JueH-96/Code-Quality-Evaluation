def solve():
    n = int(input())
    c = []
    a = []
    for _ in range(n):
        ci = int(input())
        ai = list(map(int, input().split()))
        c.append(ci)
        a.append(ai)
    x = int(input())

    bettors = []
    for i in range(n):
        if x in a[i]:
            bettors.append(i)

    if not bettors:
        print(0)
        return

    min_bets = float('inf')
    for i in bettors:
        min_bets = min(min_bets, c[i])

    result = []
    for i in bettors:
        if c[i] == min_bets:
            result.append(i + 1)

    result.sort()
    print(len(result))
    print(*result)

solve()