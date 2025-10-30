import sys

# Read the input
N = int(sys.stdin.readline().strip())
people = []
for _ in range(N):
    name, age = sys.stdin.readline().strip().split()
    people.append((name, int(age)))

# Sort the people by age in ascending order
people.sort(key=lambda x: x[1])

# Print the names in clockwise order
for name, _ in people:
    print(name)