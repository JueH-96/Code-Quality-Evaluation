def main():
    import sys
    input_data = sys.stdin.read().split()
    N = int(input_data[0])
    P = int(input_data[1])
    Q = int(input_data[2])
    # Read dish prices
    dish_prices = list(map(int, input_data[3:3+N]))
    
    # Calculate the total cost if using the coupon
    cost_with_coupon = Q + min(dish_prices)
    
    # The minimum total amount is the lower of the regular price and coupon combination
    result = min(P, cost_with_coupon)
    
    print(result)

if __name__ == '__main__':
    main()