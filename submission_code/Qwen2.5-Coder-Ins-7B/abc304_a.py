# YOUR CODE HERE
import sys

# Read input from stdin
input = sys.stdin.read
data = input().split()

# Parse input
N = int(data[0])
people = []
for i in range(N):
    name = data[2 * i + 1]
    age = int(data[2 * i + 2])
    people.append((name, age))

# Find the youngest person
youngest_index = people.index(min(people, key=lambda x: x[1]))

# Print names in clockwise order starting from the youngest person
for i in range(N):
    print(people[(youngest_index + i) % N][0])