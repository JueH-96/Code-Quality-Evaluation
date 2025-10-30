import sys
from fractions import Fraction

# Read all input from stdin
data = sys.stdin.read().split()

# Initialize index to read from data list
idx = 0
N = int(data[idx])
idx += 1

# Store list of people with their index and success rate fraction
people = []
for i in range(N):
    A = int(data[idx])
    B = int(data[idx + 1])
    idx += 2
    S = A + B  # Total tosses
    frac = Fraction(A, S)  # Success rate as a fraction
    people.append((i + 1, frac))  # Store 1-based index and fraction

# Sort the people list in descending order of success rate
# Python's sort is stable, so equal rates keep original ascending index order
sorted_people = sorted(people, key=lambda x: x[1], reverse=True)

# Extract the indices from the sorted list
indices = [person[0] for person in sorted_people]

# Output the indices separated by spaces
print(' '.join(map(str, indices)))