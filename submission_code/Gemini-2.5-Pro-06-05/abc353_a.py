# YOUR CODE HERE
import sys

def solve():
  """
  Reads building data and finds the first building taller than the first one.
  """
  # Read the number of buildings, N.
  try:
    n_str = sys.stdin.readline()
    if not n_str:
      return
    n = int(n_str)
    
    # Read the N heights into a list of integers.
    heights = list(map(int, sys.stdin.readline().split()))
  except (IOError, ValueError):
    # Handle potential empty input or malformed input gracefully.
    return

  # Store the height of the first building (at index 0) for comparison.
  first_building_height = heights[0]

  # Iterate through the buildings starting from the second one (at index 1).
  for i in range(1, n):
    # Check if the current building is strictly taller than the first one.
    if heights[i] > first_building_height:
      # If a taller building is found, it's the leftmost one because we
      # are iterating from left to right.
      # Print its 1-based position (index + 1).
      print(i + 1)
      # Exit the function since the answer has been found.
      return

  # If the loop completes without finding any taller building,
  # it means no such building exists. In this case, print -1.
  print(-1)

solve()