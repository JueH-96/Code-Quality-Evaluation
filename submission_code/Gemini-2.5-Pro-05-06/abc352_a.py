# Read input values for N, X, Y, Z
# N is the total number of stations.
# X is the starting station.
# Y is the destination station.
# Z is the station to check.
# The problem states N is not used in the logic, so we can assign it to _N (convention for unused variable).
_N, X, Y, Z = map(int, input().split())

# Determine the numerical lower and upper bounds of the travel segment.
# Regardless of whether the travel is inbound (X < Y) or outbound (X > Y),
# the stations visited are those between X and Y, inclusive.
# Let station_A be the smaller of X and Y, and station_B be the larger.
station_A = min(X, Y)
station_B = max(X, Y)

# Since X, Y, and Z are distinct, Z cannot be X or Y.
# If Z is on the path, it must be strictly between X and Y.
# This means station_A < Z < station_B.

if station_A < Z < station_B:
    # Z is strictly between X and Y, so it's an intermediate stop.
    print("Yes")
else:
    # Z is not strictly between X and Y.
    print("No")