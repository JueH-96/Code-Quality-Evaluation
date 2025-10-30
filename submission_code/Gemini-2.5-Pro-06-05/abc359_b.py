import sys

def solve():
  """
  This function reads the input, solves the problem, and prints the output.
  """
  # Read N from the first line of standard input.
  # N is the number of colors.
  try:
    n_str = sys.stdin.readline()
    if not n_str:
      return
    n = int(n_str)
  except (ValueError, IndexError):
    return

  # Read the 2N colors from the second line of standard input into a list.
  try:
    a = list(map(int, sys.stdin.readline().split()))
  except (ValueError, IndexError):
    return

  # Initialize a counter for colors that satisfy the condition.
  ans = 0

  # The condition is that there is exactly one person between two people
  # wearing clothes of the same color. In a 0-indexed list, this means
  # their indices differ by 2, i.e., they are at i and i+2.

  # We iterate through the list `a`. The list has 2*n elements, so its
  # indices are from 0 to 2*n - 1.
  # To safely access a[i+2], the loop for i must stop when i+2 is the
  # last index, so the maximum value for i is (2*n - 1) - 2 = 2*n - 3.
  # The `range(2 * n - 2)` will generate i from 0 to 2*n - 3.
  for i in range(2 * n - 2):
    # Check if the colors at index i and i+2 are the same.
    if a[i] == a[i+2]:
      # If they are, this color satisfies the condition.
      # Since each color appears exactly twice, this check is sufficient
      # to identify one such color.
      ans += 1

  # Print the final count.
  print(ans)

solve()