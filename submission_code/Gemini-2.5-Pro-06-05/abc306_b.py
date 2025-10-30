import sys

def solve():
  """
  Reads a sequence of 64 bits and computes the corresponding integer value.
  """
  
  # Read the space-separated integers from a single line of standard input.
  # map(int, ...) creates an iterator of integers from the input strings.
  bits = map(int, sys.stdin.readline().split())

  # Use a generator expression with sum() to calculate the result.
  # enumerate() provides the index `i` (power of 2) and the `bit` value.
  # (1 << i) calculates 2**i efficiently.
  # The expression `bit * (1 << i)` evaluates to 2**i if the bit is 1, and 0 otherwise.
  result = sum(bit * (1 << i) for i, bit in enumerate(bits))

  # Print the final integer value to standard output.
  print(result)

solve()