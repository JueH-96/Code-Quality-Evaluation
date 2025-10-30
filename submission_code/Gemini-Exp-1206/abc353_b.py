def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    
    ans = 0
    empty_seats = k
    
    i = 0
    while i < len(a):
        if empty_seats < a[i]:
            ans += 1
            empty_seats = k
        else:
            empty_seats -= a[i]
            i += 1
    
    if empty_seats < k:
        ans += 1
        
    print(ans)

solve()