# YOUR CODE HERE
import sys

def solve():
  """
  Reads three integers, sorts them, and checks if the product of the two
  smallest equals the largest.
  """
  try:
    # Read the three integers from a single line of input.
    # The `map(int, ...)` function applies the `int` conversion to each element
    # that results from `input().split()`.
    # The `list(...)` constructor converts the map object into a list.
    A = list(map(int, sys.stdin.readline().split()))
  except (IOError, ValueError):
    # This handles cases with no input or invalid input, though not
    # expected under the problem constraints.
    return

  # Sort the list to easily identify the smallest and largest numbers.
  A.sort()

  # For a permutation (B1, B2, B3) to satisfy B1 * B2 = B3, B3 must be the
  # largest number (since inputs are >= 1). So we only need to check if the
  # product of the two smaller numbers equals the largest.
  if A[0] * A[1] == A[2]:
    print("Yes")
  else:
    print("No")

solve()