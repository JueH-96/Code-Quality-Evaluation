def solve():
    n, m = map(int, input().split())
    h = list(map(int, input().split()))
    
    count = 0
    for hands in h:
        if m >= hands:
            count += 1
            m -= hands
        else:
            m = 0
    
    print(count)

solve()