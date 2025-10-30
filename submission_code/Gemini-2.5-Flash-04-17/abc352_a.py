# Read the input from standard input
# The input line contains four space-separated integers: N, X, Y, Z
n, x, y, z = map(int, input().split())

# The problem asks if station Z is stopped at *during* the travel from station X to station Y.
# "During" implies stations visited strictly between the departure from X and arrival at Y.

# There are two possible travel directions determined by X and Y:
# 1. If X < Y: The train is moving in the increasing order of station numbers (inbound).
#    The stations visited *between* X and Y are X+1, X+2, ..., Y-1.
# 2. If X > Y: The train is moving in the decreasing order of station numbers (outbound).
#    The stations visited *between* X and Y are X-1, X-2, ..., Y+1.

# Station Z is visited during the trip if it falls strictly within this sequence of stations.
# In both cases (X < Y and X > Y), the set of stations visited strictly between X and Y
# are those with station numbers k such that min(X, Y) < k < max(X, Y).
# Therefore, the train stops at station Z during the travel from X to Y if and only if
# Z is strictly between the minimum and maximum of X and Y.

# Check if Z falls strictly between min(x, y) and max(x, y).
if min(x, y) < z < max(x, y):
    # If the condition is true, Z is visited during the travel.
    print("Yes")
else:
    # Otherwise, Z is not visited during the travel.
    print("No")