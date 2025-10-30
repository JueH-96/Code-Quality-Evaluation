def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    
    empty_seats = k
    starts = 0
    
    i = 0
    while i < n:
        if empty_seats < a[i]:
            starts += 1
            empty_seats = k
        else:
            empty_seats -= a[i]
            i += 1
            
    if i == n and empty_seats < k:
        starts += 1
    
    print(starts)

solve()