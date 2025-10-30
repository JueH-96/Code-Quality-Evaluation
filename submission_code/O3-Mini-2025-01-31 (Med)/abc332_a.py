def main():
    import sys
    input = sys.stdin.readline

    # Read first input line: N, S, K
    N, S, K = map(int, input().split())
    
    # Initialize the total price
    total_price = 0
    
    # Read each product's price and quantity and add to total
    for _ in range(N):
        P, Q = map(int, input().split())
        total_price += P * Q

    # Determine shipping fee
    shipping_fee = 0 if total_price >= S else K

    # Print the sum of total price and the shipping fee
    print(total_price + shipping_fee)

if __name__ == '__main__':
    main()