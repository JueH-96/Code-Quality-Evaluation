# YOUR CODE HERE
import sys

def solve():
  """
  Reads the input, calculates the total score for problems with a score of X or less,
  and prints the result.
  """

  # Read the first line of input containing N and X.
  # N is the number of problems and X is the score threshold.
  # The input is split by spaces, and each part is converted to an integer.
  # While N is read, it's not directly used in the final calculation,
  # as the length of the scores list implicitly provides this information.
  try:
    n, x = map(int, sys.stdin.readline().split())

    # Read the second line of input containing the scores S_1, S_2, ..., S_N.
    # This creates a list of integer scores.
    scores = list(map(int, sys.stdin.readline().split()))
  except (IOError, ValueError) as e:
    # Handle potential empty lines or invalid input format, though
    # problem constraints suggest this won't happen in the contest environment.
    # For robustness, it's good practice.
    print(f"Error reading input: {e}", file=sys.stderr)
    return

  # Calculate the sum of scores that are less than or equal to X.
  # A generator expression `(s for s in scores if s <= x)` is used for this.
  # It iterates through each score `s` in the `scores` list.
  # If a score `s` meets the condition `s <= x`, it is included.
  # The built-in `sum()` function then calculates the total of these filtered scores.
  total_score = sum(s for s in scores if s <= x)

  # Print the final result to standard output.
  print(total_score)

# Execute the main logic.
solve()