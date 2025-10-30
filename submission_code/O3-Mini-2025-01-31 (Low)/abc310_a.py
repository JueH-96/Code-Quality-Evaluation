def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data: 
        return
    N = int(data[0])
    P = int(data[1])
    Q = int(data[2])
    D = list(map(int, data[3:3+N]))
    
    # Option 1: Buy the drink at regular price
    regular_price = P
    
    # Option 2: Use the coupon and add minimum dish price
    coupon_price = Q + min(D)
    
    # Minimum total cost
    print(min(regular_price, coupon_price))

if __name__ == '__main__':
    main()