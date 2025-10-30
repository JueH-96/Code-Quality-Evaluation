# YOUR CODE HERE
import sys

def solve():
  """
  Reads N, M, P and calculates the number of full moon days from day 1 to N.
  """
  # Read N, M, P from a single line of standard input.
  # N: The last day of the observation period.
  # M: The day of the first full moon.
  # P: The period, in days, between consecutive full moons.
  try:
    N, M, P = map(int, sys.stdin.readline().split())
  except (IOError, ValueError):
    # This handles potential empty lines or malformed input,
    # though problem constraints usually guarantee valid input.
    return

  # If the first full moon day (M) is after the last day of interest (N),
  # then there are no full moon days within the specified range [1, N].
  if M > N:
    print(0)
    return

  # The days of the full moon form an arithmetic progression.
  # We need to count how many terms of the sequence M, M+P, M+2P, ... are <= N.
  
  # The first full moon is on day M.
  # The number of additional full moons is the number of times the period P
  # fits into the remaining interval of (N - M) days.
  # This can be calculated using integer division: (N - M) // P.
  
  # The total count is 1 (for the first full moon on day M) plus the
  # number of additional full moons.
  count = (N - M) // P + 1

  # Print the final result to standard output.
  print(count)

if __name__ == "__main__":
  solve()