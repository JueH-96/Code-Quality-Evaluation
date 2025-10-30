import sys

def solve():
    # Read N, P, Q from the first line
    N, P, Q = map(int, sys.stdin.readline().split())

    # Read the list of D_i values from the second line
    D = list(map(int, sys.stdin.readline().split()))

    # Option 1: Pay the regular price P
    cost_regular_price = P

    # Option 2: Use the discount coupon
    # To use the coupon, Takahashi must order one of the N dishes.
    # To minimize the total cost with the coupon, he should choose the cheapest dish.
    min_dish_price = min(D)
    cost_with_coupon = Q + min_dish_price

    # The minimum total amount is the minimum of the two options
    result = min(cost_regular_price, cost_with_coupon)

    # Print the result
    print(result)

# Call the solve function
solve()