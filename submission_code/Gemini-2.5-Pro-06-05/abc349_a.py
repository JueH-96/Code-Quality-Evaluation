# YOUR CODE HERE
def solve():
  """
  Reads the input, calculates the score of the Nth person, and prints the result.
  """
  # Read the integer N. This is not used in the calculation but is part of the input format.
  try:
    N = int(input())
  except (ValueError, EOFError):
    return

  # Read the N-1 scores as a list of integers.
  try:
    A = list(map(int, input().split()))
  except (ValueError, EOFError):
    return

  # The sum of scores of all N people is an invariant and remains 0.
  # So, the score of person N is the negative of the sum of the scores of the other N-1 people.
  # We calculate the sum of the given scores and then negate it.
  result = -sum(A)

  # Print the final answer.
  print(result)

solve()