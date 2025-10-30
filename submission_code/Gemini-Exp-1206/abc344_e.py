def solve():
    n = int(input())
    a = list(map(int, input().split()))
    q = int(input())

    for _ in range(q):
        query = list(map(int, input().split()))
        if query[0] == 1:
            x, y = query[1], query[2]
            idx = a.index(x)
            a.insert(idx + 1, y)
        else:
            x = query[1]
            a.remove(x)

    print(*a)

solve()