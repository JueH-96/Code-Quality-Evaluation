# YOUR CODE HERE
def solve():
  """
  Calculates the sum of integers in a 9x9 multiplication table that are not equal to X.
  """
  # Read the integer X from standard input.
  try:
    X = int(input())
  except (ValueError, EOFError):
    return # Exit if input is invalid or empty

  # The sum of all numbers in a 9x9 multiplication table is a constant:
  # (1+2+...+9) * (1+2+...+9) = 45 * 45 = 2025.
  total_sum = 2025

  # To find the sum of numbers not equal to X, we can subtract the sum of all numbers
  # that ARE equal to X from the total sum.

  # Count occurrences of X in the grid.
  # X is at cell (i, j) if i * j = X, with 1 <= i, j <= 9.
  count_X = 0
  for i in range(1, 10):
    # If i is a factor of X, find the other factor j = X // i.
    # We need to check if j is also a valid grid index (1 to 9).
    # Since X is between 1 and 81 and i is between 1 and 9, j is always positive.
    # So, we only need to check if j is less than or equal to 9.
    if X % i == 0 and X // i <= 9:
      count_X += 1

  # Calculate the sum of all occurrences of X.
  sum_to_subtract = count_X * X

  # Calculate the final answer.
  answer = total_sum - sum_to_subtract

  # Print the answer to standard output.
  print(answer)

solve()