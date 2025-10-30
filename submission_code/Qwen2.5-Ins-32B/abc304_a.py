import sys

N = int(input())
people = []

for _ in range(N):
    S, A = input().split()
    A = int(A)
    people.append((A, S))

# Sort people by age
people.sort()

# Find the index of the youngest person
youngest_index = people.index(min(people))

# Rotate the list so that the youngest person is first
people = people[youngest_index:] + people[:youngest_index]

# Print the names in the new order
for _, name in people:
    print(name)