# YOUR CODE HERE
def solve():
  """
  Calculates the distance between two points on a line.
  """
  # Read the two points from standard input
  p, q = input().split()

  # A dictionary mapping each point to its coordinate on the line,
  # assuming point A is at position 0.
  # The coordinates are calculated based on the given adjacent distances:
  # A: 0
  # B: 0 + 3 = 3
  # C: 3 + 1 = 4
  # D: 4 + 4 = 8
  # E: 8 + 1 = 9
  # F: 9 + 5 = 14
  # G: 14 + 9 = 23
  positions = {
      'A': 0,
      'B': 3,
      'C': 4,
      'D': 8,
      'E': 9,
      'F': 14,
      'G': 23
  }

  # Get the coordinates of the two input points
  pos_p = positions[p]
  pos_q = positions[q]

  # The distance is the absolute difference between their coordinates
  distance = abs(pos_p - pos_q)

  # Print the result to standard output
  print(distance)

solve()