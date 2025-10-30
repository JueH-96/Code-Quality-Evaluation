import sys

def solve(N, M, P, L, D):
    # Sort the prices in descending order
    P.sort(reverse=True)
    
    # Sort the coupons in descending order of discount
    coupons = sorted(zip(L, D), key=lambda x: x[1], reverse=True)
    
    # Initialize the total cost
    total_cost = 0
    
    # Initialize the index for the coupons
    coupon_index = 0
    
    # Iterate over the prices
    for price in P:
        # If there are still coupons left and the current price is greater than or equal to the limit of the current coupon
        if coupon_index < M and price >= coupons[coupon_index][0]:
            # Apply the coupon
            total_cost += price - coupons[coupon_index][1]
            # Move to the next coupon
            coupon_index += 1
        else:
            # If no coupon can be applied, add the full price to the total cost
            total_cost += price
    
    return total_cost

def main():
    # Read the input
    N, M = map(int, sys.stdin.readline().split())
    P = list(map(int, sys.stdin.readline().split()))
    L = list(map(int, sys.stdin.readline().split()))
    D = list(map(int, sys.stdin.readline().split()))
    
    # Solve the problem
    result = solve(N, M, P, L, D)
    
    # Print the result
    print(result)

if __name__ == "__main__":
    main()