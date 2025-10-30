# YOUR CODE HERE
def train_stops(N, X, Y, Z):
    if X < Y:  # Inbound train
        return "Yes" if X < Z < Y else "No"
    else:  # Outbound train
        return "Yes" if Y < Z < X else "No"

# Read input
import sys
input = sys.stdin.read
N, X, Y, Z = map(int, input().strip().split())

# Output the result
print(train_stops(N, X, Y, Z))