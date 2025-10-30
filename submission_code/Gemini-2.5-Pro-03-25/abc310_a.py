# YOUR CODE HERE
import sys

def solve():
    """
    Reads input N, P, Q and the list of dish prices D.
    Calculates the minimum cost to get the AtCoder Drink.
    Prints the minimum cost to standard output.
    """
    # Read N, P, Q from the first line
    line1 = sys.stdin.readline().split()
    # n = int(line1[0]) # N is the number of dishes, not strictly needed after reading D
    p = int(line1[1]) # Regular price of the drink
    q = int(line1[2]) # Discounted price of the drink

    # Read the dish prices from the second line
    line2 = sys.stdin.readline().split()
    d = list(map(int, line2)) # List of dish prices

    # Option 1: Pay the regular price P
    cost_option1 = p

    # Option 2: Use the coupon
    # To minimize the cost with the coupon, choose the cheapest dish
    # The constraint N >= 1 ensures the list d is not empty
    min_dish_price = min(d)
    # Total cost with coupon = discounted drink price + minimum dish price
    cost_option2 = q + min_dish_price

    # The minimum total amount is the minimum of the two options
    min_total_cost = min(cost_option1, cost_option2)

    # Print the result
    print(min_total_cost)

# Execute the solve function
solve()

# END OF YOUR CODE HERE