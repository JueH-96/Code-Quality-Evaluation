def solve():
    n = int(input())
    qr = []
    for _ in range(n):
        q, r = map(int, input().split())
        qr.append((q, r))
    
    q = int(input())
    for _ in range(q):
        t, d = map(int, input().split())
        t -= 1
        
        qt, rt = qr[t]
        
        if d % qt == rt:
            print(d)
        elif d % qt < rt:
            print(d + (rt - d % qt))
        else:
            print(d + (qt - d % qt) + rt)

solve()