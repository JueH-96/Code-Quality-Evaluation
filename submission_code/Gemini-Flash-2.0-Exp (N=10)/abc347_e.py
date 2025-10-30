def solve():
    n, q = map(int, input().split())
    x = list(map(int, input().split()))
    
    a = [0] * n
    s = set()
    
    for xi in x:
        if xi in s:
            s.remove(xi)
        else:
            s.add(xi)
        
        for j in s:
            a[j-1] += len(s)
            
    print(*a)

solve()