# YOUR CODE HERE
def solve():
    # Read N, P, Q from the first line of input
    # N is the number of dishes
    # P is the regular price of the drink
    # Q is the discounted price of the drink
    N, P, Q = map(int, input().split())

    # Read the prices of the N dishes from the second line of input
    # D is a list of dish prices
    D = list(map(int, input().split()))

    # Scenario 1: Buy the drink at the regular price P
    cost_scenario_1 = P

    # Scenario 2: Use the discount coupon
    # Pay Q for the drink + price of the cheapest dish
    # Find the minimum dish price
    min_dish_price = min(D) # Since N >= 1, D is never empty.
    cost_scenario_2 = Q + min_dish_price

    # The minimum total amount is the smaller of the two scenario costs
    min_total_payment = min(cost_scenario_1, cost_scenario_2)

    # Print the result
    print(min_total_payment)

if __name__ == '__main__':
    solve()