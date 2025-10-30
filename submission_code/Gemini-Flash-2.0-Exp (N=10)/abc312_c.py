def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    all_prices = sorted(list(set(a + b)))
    
    for price in all_prices:
        sellers_at_price = sum(1 for x in a if x <= price)
        buyers_at_price = sum(1 for x in b if x >= price)
        
        if sellers_at_price >= buyers_at_price:
            print(price)
            return
    
    max_b = max(b) if b else 0
    
    print(max(a) + 1)

solve()