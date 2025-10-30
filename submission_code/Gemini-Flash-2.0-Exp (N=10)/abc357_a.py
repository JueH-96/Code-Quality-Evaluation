def solve():
    n, m = map(int, input().split())
    h = list(map(int, input().split()))
    
    count = 0
    for hands in h:
        if m >= hands:
            m -= hands
            count += 1
        else:
            m = 0
            break
    print(count)

solve()