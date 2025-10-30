def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    
    for i in range(m):
        c = 0
        balls = a[b[i]]
        a[b[i]] = 0
        while balls > 0:
            c += 1
            a[(b[i] + c) % n] += 1
            balls -= 1
    
    print(*a)

solve()