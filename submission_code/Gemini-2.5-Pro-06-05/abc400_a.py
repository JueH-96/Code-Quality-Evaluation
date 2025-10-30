# YOUR CODE HERE
import sys

def solve():
  """
  This function solves the problem of finding the number of columns B
  for a rectangular formation of 400 people given the number of rows A.
  """
  try:
    # Read the number of rows, A, from standard input.
    A = int(sys.stdin.readline())
  except (ValueError, IndexError):
    # This handles cases of empty or non-integer input, though
    # the problem constraints guarantee a valid integer.
    return

  # The total number of people is 400.
  # The formation is A rows by B columns, so A * B = 400.
  # We need to find a positive integer B.

  # An integer B exists if and only if 400 is perfectly divisible by A.
  # We use the modulo operator (%) to check for divisibility.
  if 400 % A == 0:
    # If A is a divisor of 400, calculate B.
    # We use integer division (//) to get the integer result.
    B = 400 // A
    print(B)
  else:
    # If A is not a divisor of 400, no such integer B exists.
    print(-1)

# Execute the solution
solve()