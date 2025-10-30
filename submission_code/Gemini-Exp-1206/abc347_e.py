def solve():
    n, q = map(int, input().split())
    queries = list(map(int, input().split()))
    
    a = [0] * n
    s = set()
    
    for x in queries:
        if x in s:
            s.remove(x)
        else:
            s.add(x)
        
        for j in range(n):
            if (j + 1) in s:
                a[j] += len(s)
    
    print(*a)

solve()