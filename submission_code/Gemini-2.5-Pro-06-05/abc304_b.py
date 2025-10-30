# YOUR CODE HERE
def solve():
  """
  Reads an integer N from standard input, approximates it based on its magnitude
  according to the problem rules, and prints the result to standard output.
  """
  try:
    N = int(input())
  except (IOError, ValueError):
    # Handle potential errors during input, though not expected by problem constraints.
    return

  # The problem specifies different truncation rules based on the range of N.
  # An if/elif/else chain is a clear and efficient way to implement this.

  # If N is up to 3 digits (0 to 999)
  if N < 1000:
    # Print N as it is.
    print(N)
  # If N is 4 digits (1,000 to 9,999)
  elif N < 10000:
    # Truncate the ones digit. This is equivalent to rounding down to the nearest 10.
    print((N // 10) * 10)
  # If N is 5 digits (10,000 to 99,999)
  elif N < 100000:
    # Truncate the tens digit and below. Round down to the nearest 100.
    print((N // 100) * 100)
  # If N is 6 digits (100,000 to 999,999)
  elif N < 1000000:
    # Truncate the hundreds digit and below. Round down to the nearest 1,000.
    print((N // 1000) * 1000)
  # If N is 7 digits (1,000,000 to 9,999,999)
  elif N < 10000000:
    # Truncate the thousands digit and below. Round down to the nearest 10,000.
    print((N // 10000) * 10000)
  # If N is 8 digits (10,000,000 to 99,999,999)
  elif N < 100000000:
    # Truncate the ten-thousands digit and below. Round down to the nearest 100,000.
    print((N // 100000) * 100000)
  # If N is 9 digits (100,000,000 to 999,999,999)
  else:
    # Truncate the hundred-thousands digit and below. Round down to the nearest 1,000,000.
    print((N // 1000000) * 1000000)

if __name__ == "__main__":
  solve()