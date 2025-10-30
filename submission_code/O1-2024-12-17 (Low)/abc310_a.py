def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, P, Q = map(int, data[:3])
    dishes = list(map(int, data[3:]))

    # Option 1: Not using the coupon
    cost_without_coupon = P

    # Option 2: Using the coupon, we must pay Q + min(dishes)
    cost_with_coupon = Q + min(dishes)

    # Print the minimum total cost
    print(min(cost_without_coupon, cost_with_coupon))

# Do not remove the following function call
main()