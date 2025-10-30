def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    ans = 0
    for i in range(n):
        diff = (b[i] - a[i]) % m
        if diff == 0:
            continue
        
        if i > 0 and i < n - 1:
            if (a[i] + diff) % m == a[i-1] or (a[i] + diff) % m == a[i+1]:
                print("-1")
                return
        elif i == 0 and n > 1:
            if (a[i] + diff) % m == a[i+1]:
                print("-1")
                return
        elif i == n - 1 and n > 1:
            if (a[i] + diff) % m == a[i-1]:
                print("-1")
                return
        
        ans += diff
        
    print(ans)

solve()