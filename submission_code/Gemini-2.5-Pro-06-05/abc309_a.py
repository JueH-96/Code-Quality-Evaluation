# YOUR CODE HERE
import sys

def solve():
  """
  Reads two integers A and B and determines if they are horizontally adjacent
  on a 3x3 board.
  """
  try:
    # Read the two integers from a single line of input
    A, B = map(int, sys.stdin.readline().split())

    # For A and B to be horizontally adjacent (given A < B):
    # 1. B must be exactly 1 greater than A.
    # 2. A cannot be on the rightmost column (i.e., A cannot be 3 or 6).
    #    This can be checked with the condition A % 3 != 0.
    #    (1%3=1, 2%3=2, 4%3=1, 5%3=2, 7%3=1, 8%3=2)
    #    (3%3=0, 6%3=0)
    # The number 9 is not a concern for A, since A < B <= 9.

    if B - A == 1 and A % 3 != 0:
      print("Yes")
    else:
      print("No")

  except (IOError, ValueError) as e:
    # Handle potential errors with input reading, although problem constraints
    # should prevent this.
    sys.stderr.write(f"An error occurred: {e}
")

solve()