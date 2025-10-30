def solve():
    import sys
    data = sys.stdin.read().strip().split()
    
    N, P, Q = map(int, data[:3])
    D = list(map(int, data[3:]))

    # Regular price total is just P.
    regular_price = P

    # Coupon price requires ordering at least one dish; we pick the cheapest dish.
    coupon_price = Q + min(D)

    # We want the minimum of these two options.
    print(min(regular_price, coupon_price))

def main():
    solve()

if __name__ == "__main__":
    main()