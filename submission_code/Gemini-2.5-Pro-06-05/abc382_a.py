# YOUR CODE HERE
import sys

def solve():
  """
  This function reads the input, solves the problem, and prints the output.
  """
  # Read the first line of input to get N and D.
  # N is the total number of boxes.
  # D is the number of days (and thus cookies to be eaten).
  try:
    line1 = sys.stdin.readline()
    if not line1:
      return
    # We don't actually need the value of N for this solution, so we can
    # assign it to a throwaway variable like '_'.
    _, D = map(int, line1.split())
  except (IOError, ValueError):
    # Handle potential empty lines or malformed input
    return

  # Read the second line of input, which is the string S.
  # .strip() removes any trailing newline characters.
  try:
    S = sys.stdin.readline().strip()
  except IOError:
    return

  # The total number of empty boxes after D days is the sum of:
  # 1. The number of boxes that were already empty at the start.
  # 2. The number of boxes that become empty after their cookie is eaten.

  # 1. Count the number of initially empty boxes (represented by '.').
  initially_empty_count = S.count('.')

  # 2. Over D days, D cookies are eaten. This vacates D boxes.
  # So, D more boxes become empty.

  # The final number of empty boxes is the sum of these two quantities.
  final_empty_count = initially_empty_count + D

  # Print the result to standard output.
  print(final_empty_count)

# Execute the solution
solve()