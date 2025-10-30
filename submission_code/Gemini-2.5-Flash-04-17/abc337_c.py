import sys

# Read N
N = int(sys.stdin.readline())

# Read A
# A is 0-indexed list from input
# A[i] gives information about person i+1
A = list(map(int, sys.stdin.readline().split()))

# Initialize the 'behind' array/map
# behind[j] will store the person k such that person k is directly behind person j
# Use 1-based indexing for people (1 to N)
# Size N+1. Index 0 is unused. We initialize with 0, which also acts as a sentinel
# indicating that no one is behind that person (used for the last person in line).
behind = [0] * (N + 1)

# Find the person at the front and build the 'behind' mapping
front_person = -1
for i in range(N):
    person_in_front = A[i]
    person_id = i + 1 # The person described by A[i] is person i+1

    if person_in_front == -1:
        # Person i+1 is at the front
        front_person = person_id
    else:
        # Person i+1 is behind person_in_front (which is A[i]).
        # This means person_in_front (A[i]) is immediately followed by person i+1.
        # Store this relationship in the 'behind' map.
        # A[i] gives the person ID in front, which matches the 1-based index for `behind`
        behind[person_in_front] = person_id

# Traverse the line starting from the front person
current_person = front_person
output_list = []

# We know there are exactly N people and they form a single line.
# We can traverse the line starting from the front_person.
# The traversal follows the 'behind' links until we reach the end.
# Since there are N people, we expect the loop to run N times.
for _ in range(N):
    output_list.append(current_person)
    # Move to the person behind the current_person using the 'behind' map.
    # When we reach the last person, behind[last_person] will be 0 (the initialized value),
    # and the loop terminates after the Nth iteration.
    current_person = behind[current_person]


# Print the result
# Convert the list of integer person IDs into a space-separated string.
print(' '.join(map(str, output_list)))