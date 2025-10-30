import sys
from collections import defaultdict

# Read the number of people and insurances from input
N, M = map(int, sys.stdin.readline().split())

# Read the parent information for each person
parents = list(map(int, sys.stdin.readline().split()))
parents.insert(0, 0)  # Insert a dummy value at the beginning for 1-indexing

# Create a dictionary to hold the descendants of each person
descendants = defaultdict(set)

# Populate the descendants dictionary
for i in range(2, N + 1):
    parent = parents[i - 1]
    descendants[parent].add(i)

# Function to get all descendants within y generations
def get_descendants(person, y, current_generation=0):
    covered = {person}
    if current_generation < y:
        for child in descendants[person]:
            covered.update(get_descendants(child, y, current_generation + 1))
    return covered

# Set to hold all covered people
covered_people = set()

# Process each insurance
for _ in range(M):
    x, y = map(int, sys.stdin.readline().split())
    covered_people.update(get_descendants(x, y))

# Output the number of covered people
print(len(covered_people))