# YOUR CODE HERE
import sys

def solve():
  """
  Reads a string S and its length N, then finds the first 1-based position
  of the substring "ABC". If not found, prints -1.
  """
  # Read the length of the string, N.
  # This is part of the input format but not strictly necessary for this solution.
  try:
    _ = int(sys.stdin.readline())
  except (ValueError, IndexError):
    # Handles cases of empty input, though constraints guarantee valid input.
    return

  # Read the string S and remove any trailing newline characters.
  S = sys.stdin.readline().strip()

  # Use the built-in `find()` method to locate the first occurrence of "ABC".
  # `find()` returns the starting index (0-based) of the substring if it is found.
  # If the substring is not found, `find()` returns -1.
  result_index = S.find("ABC")

  # The problem requires the output to be a 1-based position.
  # If `result_index` is -1, it means "ABC" was not found in S.
  # In this case, we print -1 as specified.
  if result_index == -1:
    print(-1)
  else:
    # If the substring was found, its 0-based index is `result_index`.
    # To convert this to a 1-based position, we add 1.
    print(result_index + 1)

solve()