# YOUR CODE HERE
import sys

def solve():
    """
    Reads input values N, X, Y, Z and determines if station Z is visited
    during the travel from station X to station Y on the AtCoder railway line.
    """
    # Read the input values N, X, Y, Z from standard input
    # N: Total number of stations
    # X: Starting station
    # Y: Destination station
    # Z: Station to check
    n, x, y, z = map(int, sys.stdin.readline().split())

    # Takahashi travels from station X to station Y.
    # If X < Y, Takahashi takes an inbound train (stations visited: X, X+1, ..., Y).
    # If X > Y, Takahashi takes an outbound train (stations visited: X, X-1, ..., Y).
    # We need to check if station Z is visited *during* this travel.
    # Since X, Y, and Z are distinct, Z cannot be the starting station X or the destination station Y.
    # Therefore, Z must be strictly between X and Y in the sequence of visited stations.

    # Case 1: Inbound train (X < Y)
    # The visited stations strictly between X and Y are X+1, X+2, ..., Y-1.
    # We need to check if Z is in this range, i.e., X < Z < Y.

    # Case 2: Outbound train (X > Y)
    # The visited stations strictly between X and Y are X-1, X-2, ..., Y+1.
    # We need to check if Z is in this range, i.e., Y < Z < X.

    # Both cases can be combined by checking if Z lies strictly between the minimum
    # and maximum of X and Y.
    start_station = min(x, y)
    end_station = max(x, y)

    # Check if Z is strictly between start_station and end_station
    if start_station < z < end_station:
        # If Z is between X and Y, it means Z is visited during the travel.
        print("Yes")
    else:
        # Otherwise, Z is not visited during the travel.
        print("No")

# Execute the solve function to run the program
solve()