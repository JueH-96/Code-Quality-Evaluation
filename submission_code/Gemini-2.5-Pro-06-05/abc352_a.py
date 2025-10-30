# YOUR CODE HERE
def solve():
  """
  Reads the input values, determines if station Z is on the path from X to Y,
  and prints the result.
  """
  try:
    # Read the four integer inputs from a single line.
    # N is the total number of stations.
    # X is the starting station.
    # Y is the destination station.
    # Z is the station to check.
    # We use map(int, input().split()) to read all integers at once.
    N, X, Y, Z = map(int, input().split())

    # The travel from station X to station Y covers all stations
    # numerically between X and Y, regardless of whether the train is
    # inbound (X < Y) or outbound (X > Y).
    #
    # We can define the range of the journey by finding the minimum and
    # maximum of the start and end stations.
    start_station = min(X, Y)
    end_station = max(X, Y)

    # The problem asks if the train stops at station Z "during" the travel.
    # Since the problem guarantees that X, Y, and Z are distinct, Z cannot
    # be the start or end station. Thus, we only need to check if Z lies
    # strictly between the start and end points of the journey's station range.
    if start_station < Z < end_station:
      print("Yes")
    else:
      print("No")

  except (IOError, ValueError):
    # This handles potential malformed input, though not expected on
    # competitive programming platforms like AtCoder.
    pass

# Run the solution function
solve()