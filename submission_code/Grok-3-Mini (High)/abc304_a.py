import sys

# Read all input and split into a list
data = sys.stdin.read().split()

# Initialize index to read from data list
index = 0
N = int(data[index])
index += 1

# Store people as a list of tuples (name, age)
people = []
for i in range(N):
    name = data[index]
    index += 1
    age = int(data[index])
    index += 1
    people.append((name, age))

# Find the index of the youngest person (min age)
min_index = min(range(N), key=lambda i: people[i][1])

# Start index for output
start_idx = min_index

# Print names starting from the youngest in clockwise order
for k in range(N):
    pos = (start_idx + k) % N
    print(people[pos][0])