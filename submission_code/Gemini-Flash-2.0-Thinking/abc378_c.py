def solve():
    n = int(input())
    a = list(map(int, input().split()))

    b = []
    last_seen = {}

    for i in range(n):
        if a[i] in last_seen:
            b.append(last_seen[a[i]])
        else:
            b.append(-1)
        last_seen[a[i]] = i + 1

    print(*b)

solve()