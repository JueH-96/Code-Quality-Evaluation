def solve():
    n = int(input())
    a = list(map(int, input().split()))
    s = []
    t = []
    for _ in range(n - 1):
        si, ti = map(int, input().split())
        s.append(si)
        t.append(ti)

    for i in range(n - 1):
        if a[i] >= s[i]:
            transfer = a[i] // s[i]
            a[i] -= transfer * s[i]
            a[i+1] += transfer * t[i]

    print(a[n-1])

solve()