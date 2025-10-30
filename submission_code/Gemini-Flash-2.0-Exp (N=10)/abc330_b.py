def solve():
    n, l, r = map(int, input().split())
    a = list(map(int, input().split()))
    
    result = []
    for x in a:
        if x < l:
            result.append(l)
        elif x > r:
            result.append(r)
        else:
            result.append(x)
            
    print(*result)

solve()