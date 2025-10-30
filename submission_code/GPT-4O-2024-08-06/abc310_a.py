# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    P = int(data[1])
    Q = int(data[2])
    D = list(map(int, data[3:]))
    
    # Calculate the minimum cost with the coupon
    min_dish_price = min(D)
    cost_with_coupon = Q + min_dish_price
    
    # Calculate the cost without the coupon
    cost_without_coupon = P
    
    # Determine the minimum cost
    min_cost = min(cost_with_coupon, cost_without_coupon)
    
    print(min_cost)

if __name__ == "__main__":
    main()