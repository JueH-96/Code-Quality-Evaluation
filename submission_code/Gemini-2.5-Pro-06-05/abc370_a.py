# YOUR CODE HERE
import sys

def solve():
  """
  Reads two integers L and R from stdin, representing the state of Snuke's hands,
  and prints the appropriate response based on the problem's rules.
  """
  try:
    # Read a line from standard input and split it by spaces.
    # map(int, ...) converts the resulting list of strings to a list of integers.
    L, R = map(int, sys.stdin.readline().split())
  except (ValueError, IndexError):
    # This handles cases where the input is not in the expected format,
    # though problem constraints usually guarantee valid input.
    return

  # Case 1: Wants takoyaki (only left hand is raised)
  if L == 1 and R == 0:
    print("Yes")
  # Case 2: Does not want takoyaki (only right hand is raised)
  elif L == 0 and R == 1:
    print("No")
  # Case 3: Invalid state (both hands are raised or no hands are raised)
  else:
    print("Invalid")

# Run the solution function
solve()