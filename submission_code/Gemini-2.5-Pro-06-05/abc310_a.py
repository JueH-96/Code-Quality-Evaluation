def main():
  """
  Main function to solve the problem.
  """
  # Read the number of dishes N, regular price P, and coupon price Q
  # The value of N is not strictly necessary for the calculation but is part of the input format.
  N, P, Q = map(int, input().split())

  # Read the prices of the N dishes into a list
  D = list(map(int, input().split()))

  # Scenario 1: Pay the regular price P.
  # The total cost is P.
  cost_regular = P

  # Scenario 2: Use the coupon.
  # The cost is the coupon price Q plus the price of the cheapest dish.
  # Find the minimum dish price.
  min_dish_price = min(D)
  cost_with_coupon = Q + min_dish_price

  # The minimum total payment is the smaller of the two scenarios.
  answer = min(cost_regular, cost_with_coupon)

  # Print the final answer to stdout
  print(answer)

if __name__ == "__main__":
  main()