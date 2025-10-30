# YOUR CODE HERE
def solve():
  """
  Reads eight integers and checks if they meet three specific conditions.
  """
  try:
    # Read the eight integers from a single line of input.
    S = list(map(int, input().split()))
  except (ValueError, IndexError):
    # This path handles malformed input, though problem constraints suggest it won't occur.
    print("No")
    return

  # Condition 1: The sequence is monotonically non-decreasing (S1 <= S2 <= ... <= S8).
  # We check this by comparing the list to its sorted version.
  is_non_decreasing = (S == sorted(S))

  # Condition 2: All numbers are between 100 and 675, inclusive.
  # We use `all()` with a generator expression for a concise check.
  is_in_range = all(100 <= x <= 675 for x in S)

  # Condition 3: All numbers are multiples of 25.
  # We use `all()` again to verify this property for every number.
  is_multiple_of_25 = all(x % 25 == 0 for x in S)

  # The final answer is "Yes" if and only if all three conditions are true.
  if is_non_decreasing and is_in_range and is_multiple_of_25:
    print("Yes")
  else:
    print("No")

solve()