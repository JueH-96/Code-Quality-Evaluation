# YOUR CODE HERE
import sys

def solve():
  """
  Reads input, processes the string, and prints the result.
  """
  # Read the first line containing N, c1, and c2.
  # N is the length of the string S, but we can infer this from the string
  # itself, so we don't need to use the variable N. We read it to correctly
  # parse the input line.
  try:
    _n, c1, c2 = sys.stdin.readline().strip().split()
  except (IOError, ValueError):
    # This handles potential empty lines or malformed input,
    # which is good practice but not expected given the problem constraints.
    return

  # Read the string S from the second line of standard input.
  # The .strip() method removes any leading/trailing whitespace, including the newline character.
  s = sys.stdin.readline().strip()

  # Construct the resulting string using a generator expression inside ''.join().
  # This is an efficient and Pythonic way to build a new string from an iterable.
  # For each character `char` in the input string `s`:
  #  - The expression `char if char == c1 else c2` evaluates to `char` if the condition is true.
  #  - Otherwise, it evaluates to `c2`.
  result_string = "".join(char if char == c1 else c2 for char in s)

  # Print the final string to standard output.
  print(result_string)

# This block ensures the solve function is called when the script is executed.
if __name__ == "__main__":
  solve()