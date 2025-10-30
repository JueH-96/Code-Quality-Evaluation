import sys

def solve():
  """
  Reads a positive integer N and prints a string of N zeros and N+1 ones
  where 0 and 1 alternate.
  """
  try:
    # Read the integer N from standard input.
    N = int(sys.stdin.readline())
  except (ValueError, IndexError):
    # Handle potential empty input or invalid format, though constraints say it's a valid integer.
    return

  # The required string has N zeros and N+1 ones, alternating.
  # This pattern can be formed by repeating "10" N times and appending a final "1".
  # For example, if N=4, this becomes "10" * 4 + "1" = "10101010" + "1" = "101010101".
  # This string correctly contains 4 zeros and 5 ones.
  result = "10" * N + "1"
  
  # Print the result to standard output.
  print(result)

solve()