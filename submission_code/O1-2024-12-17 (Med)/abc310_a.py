def main():
    import sys
    data = sys.stdin.read().strip().split()
    
    N = int(data[0])
    P = int(data[1])
    Q = int(data[2])
    D = list(map(int, data[3:]))

    # Regular cost: Just buy the drink
    regular_cost = P
    
    # Coupon cost: Buy the drink at Q yen plus the cheapest dish
    coupon_cost = Q + min(D)
    
    # Print the minimum of the two costs
    print(min(regular_cost, coupon_cost))

# Call the main function
if __name__ == "__main__":
    main()