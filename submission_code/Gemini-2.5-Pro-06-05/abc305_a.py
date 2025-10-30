# YOUR CODE HERE
import sys

def solve():
  """
  Reads Takahashi's position N and calculates the nearest water station.
  """
  # Read the integer from standard input.
  # The int() function converts the string input to an integer.
  N = int(sys.stdin.readline())

  # The problem is to find the nearest multiple of 5 to N.
  # This can be solved by dividing N by 5, rounding the result to the nearest
  # integer, and then multiplying by 5.
  #
  # Example 1: N = 53
  #   1. N / 5 = 53 / 5 = 10.6
  #   2. round(10.6) = 11
  #   3. 11 * 5 = 55
  #
  # Example 2: N = 21
  #   1. N / 5 = 21 / 5 = 4.2
  #   2. round(4.2) = 4
  #   3. 4 * 5 = 20
  #
  # Python's built-in `round()` function is sufficient here. The problem's
  # constraints ensure N is an integer, so N/5 will never be exactly
  # halfway between two integers (e.g., 10.5), which simplifies rounding.
  nearest_station = round(N / 5) * 5

  # Print the result to standard output.
  print(nearest_station)

solve()