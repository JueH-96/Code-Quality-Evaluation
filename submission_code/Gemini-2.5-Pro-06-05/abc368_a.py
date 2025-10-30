# YOUR CODE HERE
def solve():
  """
  Reads the input, performs the card operation, and prints the result.
  """
  # Read N (number of cards) and K (cards to move) from standard input.
  # The input values are on a single line, separated by a space.
  try:
    n, k = map(int, input().split())
  except (ValueError, IndexError):
    # This handles cases of empty input lines or malformed input.
    return

  # Read the list of integers A representing the cards from top to bottom.
  cards = list(map(int, input().split()))

  # The problem is to move the bottom K cards to the top.
  # This is a right rotation of the list by K positions.
  # Python's list slicing provides a concise way to do this.

  # `cards[-k:]` creates a new list with the last k elements.
  # `cards[:-k]` creates a new list with all elements except the last k.
  new_order = cards[-k:] + cards[:-k]

  # Print the elements of the new list, separated by spaces.
  # The `*` operator unpacks the list elements into individual arguments for print().
  print(*new_order)

# Run the solution
solve()