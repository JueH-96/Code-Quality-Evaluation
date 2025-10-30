def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    count = 0
    while sum(1 for x in a if x > 0) > 1:
        a.sort(reverse=True)
        a[0] = max(0, a[0] - 1)
        a[1] = max(0, a[1] - 1)
        count += 1
    
    print(count)

solve()