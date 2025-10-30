# YOUR CODE HERE
def solve():
  """
  Reads an integer B from stdin and finds a positive integer A such that A^A = B.
  Prints A if found, otherwise prints -1.
  """
  try:
    # Read the integer B from standard input.
    B = int(input())
  except (ValueError, EOFError):
    # This handles cases of non-integer or empty input, though problem
    # constraints imply the input will be a valid integer.
    print(-1)
    return

  # The function f(A) = A^A grows very rapidly. The value for A=16 exceeds
  # the maximum possible B (10^18). Thus, we only need to check A up to 15.
  # We loop up to 16 for safety.
  for A in range(1, 17):
    # Calculate A to the power of A. Python's integers handle large numbers,
    # so overflow is not an issue.
    val = A ** A

    if val == B:
      # If we find a match, print A and break the loop.
      print(A)
      break
  else:
    # The 'else' block executes only if the for loop completes without a 'break'.
    # This means no solution was found in the tested range.
    print(-1)

if __name__ == "__main__":
  solve()