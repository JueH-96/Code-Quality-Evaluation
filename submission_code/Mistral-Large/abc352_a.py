import sys

input = sys.stdin.read
data = input().split()

N = int(data[0])
X = int(data[1])
Y = int(data[2])
Z = int(data[3])

# Determine the direction of travel
if X < Y:
    # Inbound train
    stations = list(range(X + 1, Y + 1))
else:
    # Outbound train
    stations = list(range(X - 1, Y - 1, -1))

# Check if station Z is in the list of stations the train will stop at
if Z in stations:
    print("Yes")
else:
    print("No")