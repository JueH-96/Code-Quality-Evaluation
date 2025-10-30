import sys

# Read the number of people
N = int(sys.stdin.readline())

# Store the name and age for each person
# We use a list of tuples (name, age)
people = []
for _ in range(N):
    s, a = sys.stdin.readline().split()
    people.append((s, int(a)))

# Find the index of the youngest person
# Initialize minimum age to a very large value
min_age = float('inf')
youngest_index = -1

for i in range(N):
    if people[i][1] < min_age:
        min_age = people[i][1]
        youngest_index = i

# Print the names starting from the youngest person
# in clockwise order.
# The people are indexed 0 to N-1 in the list, corresponding to persons 1 to N.
# Starting from youngest_index, the next person in clockwise order is at (youngest_index + 1) % N,
# then (youngest_index + 2) % N, and so on.
# The i-th person in the output sequence (0-indexed) is at index (youngest_index + i) % N
# in the original 'people' list.
for i in range(N):
    current_index = (youngest_index + i) % N
    print(people[current_index][0])