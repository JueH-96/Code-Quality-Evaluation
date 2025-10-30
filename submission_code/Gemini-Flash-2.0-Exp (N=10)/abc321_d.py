def solve():
    n, m, p = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    
    total_price = 0
    for main_dish_price in a:
        for side_dish_price in b:
            total_price += min(main_dish_price + side_dish_price, p)
    print(total_price)

solve()