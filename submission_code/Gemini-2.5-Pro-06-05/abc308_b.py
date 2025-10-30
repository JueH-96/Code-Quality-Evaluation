# YOUR CODE HERE
def main():
  """
  Reads sushi plate information and calculates the total price.
  """
  # Read N (number of plates eaten) and M (number of special colors).
  try:
    N, M = map(int, input().split())
  except (IOError, ValueError):
    # Handles cases where input might be empty, though not expected in this problem.
    return

  # Read the list of colors for the N plates eaten.
  C = input().split()

  # Read the list of M special colors.
  D = input().split()

  # Read the list of M+1 prices.
  P = list(map(int, input().split()))

  # The first price, P[0], is the default price for any color not in D.
  default_price = P[0]

  # The subsequent prices P[1]...P[M] correspond to the special colors D.
  # We create a dictionary to map special colors to their prices for efficient lookup.
  # zip() pairs each color in D with its corresponding price from P[1:].
  price_map = dict(zip(D, P[1:]))

  # Calculate the total price using a generator expression within sum().
  # For each color in C, we get its price from price_map.
  # If the color is not in the map, we use the default_price.
  total_amount = sum(price_map.get(color, default_price) for color in C)

  # Print the final calculated total amount.
  print(total_amount)

if __name__ == "__main__":
  main()