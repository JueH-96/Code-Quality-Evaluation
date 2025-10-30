def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    for x in sorted(list(set(a + b))):
        sellers = sum(1 for price in a if price <= x)
        buyers = sum(1 for price in b if price >= x)
        if sellers >= buyers:
            print(x)
            return
    
    print(max(b) + 1)

solve()