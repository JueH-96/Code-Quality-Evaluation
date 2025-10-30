# YOUR CODE HERE
N = int(input())
people = []

for _ in range(N):
    name, age = input().split()
    people.append((name, int(age)))

# Sort people by age in ascending order
people.sort(key=lambda x: x[1])

# Find the index of the youngest person
youngest_index = 0

# Reorder the list to start from the youngest person
reordered = people[youngest_index:] + people[:youngest_index]

# Print names in clockwise order
for person in reordered:
    print(person[0])