def main():
    import sys

    data = sys.stdin.read().strip().split()
    N, S, K = map(int, data[:3])
    prices_quantities = data[3:]
    
    total_price = 0
    
    idx = 0
    for _ in range(N):
        P = int(prices_quantities[idx])
        Q = int(prices_quantities[idx + 1])
        idx += 2
        total_price += P * Q

    shipping_fee = 0 if total_price >= S else K
    print(total_price + shipping_fee)

if __name__ == "__main__":
    main()