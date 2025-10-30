# YOUR CODE HERE
import sys

input = sys.stdin.read().split()
N = int(input[0])
people = []

for i in range(N):
    name = input[2 * i + 1]
    age = int(input[2 * i + 2])
    people.append((name, age))

# Find the youngest person
youngest_person = min(people, key=lambda x: x[1])
youngest_index = people.index(youngest_person)

# Rotate the list to start from the youngest person
rotated_people = people[youngest_index:] + people[:youngest_index]

# Print the names in the rotated order
for person in rotated_people:
    print(person[0])