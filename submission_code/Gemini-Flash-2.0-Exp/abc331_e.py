def solve():
    n, m, l = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    
    forbidden = set()
    for _ in range(l):
        c, d = map(int, input().split())
        forbidden.add((c, d))
    
    max_price = 0
    found = False
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if (i, j) not in forbidden:
                price = a[i-1] + b[j-1]
                max_price = max(max_price, price)
                found = True
    
    print(max_price)

solve()