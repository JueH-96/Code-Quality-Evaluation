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
        trade_count = min(a[i] // s[i], 10**9)
        a[i+1] += trade_count * t[i]
        a[i] -= trade_count * s[i]
        
    print(a[n-1])

solve()