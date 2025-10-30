def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, S, K = map(int, data[:3])
    idx = 3
    total_price = 0

    for _ in range(N):
        P, Q = map(int, data[idx:idx+2])
        total_price += P * Q
        idx += 2

    shipping_fee = 0 if total_price >= S else K
    print(total_price + shipping_fee)

# Do not forget to call main()
if __name__ == "__main__":
    main()