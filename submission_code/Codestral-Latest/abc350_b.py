# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
Q = int(data[1])
T = list(map(int, data[2:]))

# Initialize a list to keep track of the state of each hole
holes = [True] * N  # True means there is a tooth, False means the hole is empty

for i in range(Q):
    hole = T[i] - 1  # Convert to 0-based index
    holes[hole] = not holes[hole]  # Toggle the state of the hole

# Count the number of teeth
teeth_count = holes.count(True)

print(teeth_count)