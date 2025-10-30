def solve():
    n, m = map(int, input().split())
    c = input().split()
    d = input().split()
    prices = list(map(int, input().split()))
    
    p0 = prices[0]
    price_map = {}
    for i in range(m):
        price_map[d[i]] = prices[i+1]
    
    total_price = 0
    for color in c:
        if color in price_map:
            total_price += price_map[color]
        else:
            total_price += p0
    
    print(total_price)

solve()