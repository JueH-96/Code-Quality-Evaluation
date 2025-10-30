# YOUR CODE HERE
import sys

def solve():
  """
  Reads a real number string from standard input, formats it according to the
  problem specification, and prints the result to standard output.
  """
  # Read the input string. Using sys.stdin.readline().strip() is a common
  # and efficient way to read a single line of input.
  x_str = sys.stdin.readline().strip()

  # The problem requires removing trailing zeros and then a possible trailing
  # decimal point. This can be achieved by chaining string `rstrip` methods.
  #
  # Example 1: x_str = "12.340"
  # "12.340".rstrip('0') -> "12.34"
  # "12.34".rstrip('.')  -> "12.34" (no change)
  # Result: "12.34"
  #
  # Example 2: x_str = "5.000"
  # "5.000".rstrip('0') -> "5."
  # "5.".rstrip('.')    -> "5"
  # Result: "5"
  #
  # This one-liner is efficient, readable, and directly solves the problem
  # without unnecessary type conversions.
  print(x_str.rstrip('0').rstrip('.'))

if __name__ == "__main__":
  solve()