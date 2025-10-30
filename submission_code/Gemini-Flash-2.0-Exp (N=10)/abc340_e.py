def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    for bi in b:
        c = 0
        hand = a[bi]
        a[bi] = 0
        while hand > 0:
            c += 1
            a[(bi + c) % n] += 1
            hand -= 1
    
    print(*a)

solve()