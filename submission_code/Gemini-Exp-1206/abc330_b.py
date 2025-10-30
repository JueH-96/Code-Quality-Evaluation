def solve():
    n, l, r = map(int, input().split())
    a = list(map(int, input().split()))
    
    result = []
    for ai in a:
        if ai < l:
            result.append(l)
        elif ai > r:
            result.append(r)
        else:
            result.append(ai)
    
    print(*result)

solve()