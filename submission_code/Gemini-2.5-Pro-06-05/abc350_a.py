# YOUR CODE HERE
def solve():
  """
  Reads a string S and determines if it's a valid AtCoder contest abbreviation.
  """
  # Read the input string from standard input.
  S = input()

  # The problem guarantees the first three characters are "ABC" and the last
  # three are digits. We extract the numeric part.
  # S[3:] gives the substring from index 3 to the end.
  number_str = S[3:]

  # Convert the numeric string to an integer.
  # No error handling is needed as the input format is guaranteed.
  n = int(number_str)

  # A contest abbreviation is valid if the number `n` is in the range [1, 349]
  # and is not equal to 316.
  # This can be checked with a single conditional statement.
  if 1 <= n <= 349 and n != 316:
    print("Yes")
  else:
    print("No")

# Run the solution
solve()