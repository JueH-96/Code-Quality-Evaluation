# YOUR CODE HERE
import sys

def solve():
  """
  This function solves the problem of printing pi to the N-th decimal place.
  """
  # The value of pi to the 100-th decimal place, stored as a string
  # to preserve full precision.
  pi_string = "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"

  # Read the integer N from standard input.
  try:
    N = int(sys.stdin.readline())
  except (ValueError, IndexError):
    # This handles cases with invalid input, though problem constraints guarantee valid input.
    return

  # To truncate pi to N decimal places, we need to take the integer part '3',
  # the decimal point '.', and the first N decimal digits.
  # The total length of the resulting string will be 1 (for '3') + 1 (for '.') + N = N + 2.
  # We can obtain this by slicing the pi_string from the beginning up to the (N+2)-th character.
  # String slicing s[:k] extracts the first k characters.
  truncated_pi = pi_string[:N + 2]

  # Print the final truncated string to standard output.
  print(truncated_pi)

solve()