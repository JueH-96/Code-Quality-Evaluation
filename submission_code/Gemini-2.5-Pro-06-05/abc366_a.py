# YOUR CODE HERE
import sys

def solve():
  """
  Solves the AtCoder City election problem.
  """
  try:
    # Read the single line of input values N, T, and A
    N, T, A = map(int, sys.stdin.readline().split())
  except (IOError, ValueError):
    # Handle potential empty lines or invalid input, though problem constraints suggest this won't happen.
    return

  # The total number of votes N is an odd number.
  # To win, a candidate needs more than N/2 votes.
  # The minimum integer number of votes to guarantee a win is floor(N/2) + 1.
  # In Python, this is calculated using integer division: (N // 2) + 1.
  votes_to_win = (N // 2) + 1

  # The outcome is decided if one candidate has already secured enough votes to win.
  # We check if either Takahashi's votes (T) or Aoki's votes (A) have met this threshold.
  if T >= votes_to_win or A >= votes_to_win:
    # If a candidate has already secured a majority, the outcome is decided.
    print("Yes")
  else:
    # If neither candidate has won yet, the remaining votes could potentially
    # swing the election to either side. Therefore, the outcome is not decided.
    print("No")

solve()