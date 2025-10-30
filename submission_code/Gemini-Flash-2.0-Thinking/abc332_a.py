def solve():
    n, s, k = map(int, input().split())
    total_price = 0
    for _ in range(n):
        p, q = map(int, input().split())
        total_price += p * q

    shipping_fee = k if total_price < s else 0
    total_amount = total_price + shipping_fee
    print(total_amount)

solve()