def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    P = int(data[1])
    Q = int(data[2])
    D = list(map(int, data[3:]))
    
    # Calculate the minimum cost without using the coupon
    min_cost_without_coupon = P
    
    # Calculate the minimum cost using the coupon
    min_dish_price = min(D)
    min_cost_with_coupon = Q + min_dish_price
    
    # Determine the minimum total cost
    min_total_cost = min(min_cost_without_coupon, min_cost_with_coupon)
    
    # Output the result
    print(min_total_cost)