def solve():
    n, m, p = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    total_price = 0
    for main_dish_price in a:
        for side_dish_price in b:
            set_meal_price = min(main_dish_price + side_dish_price, p)
            total_price += set_meal_price
    print(total_price)

if __name__ == "__main__":
    solve()