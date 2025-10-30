# YOUR CODE HERE
import sys

# Read the number of people
N = int(sys.stdin.readline())

# Read the list of person numbers in order of position
# P_list[i] is the person number at the (i+1)-th position from the front (1-based position)
# Using 0-based indexing for list positions in Python, P_list[i] is at index i.
# P_list has N elements.
P_list = list(map(int, sys.stdin.readline().split()))

# Create a dictionary to map each person number to their 0-indexed position in the list P_list
# This allows for O(1) average time lookup of a person's position given their number
# Key: person number (1 to N)
# Value: 0-indexed position (0 to N-1)
person_to_position = {}
# Iterate through the list P_list using enumerate to get both the index (position) and the value (person number)
for position, person_number in enumerate(P_list):
    person_to_position[person_number] = position

# Read the number of queries
Q = int(sys.stdin.readline())

# Process each query
for _ in range(Q):
    # Read the two person numbers for the current query
    # These are the numbers A_i and B_i from the problem description
    person_A, person_B = map(int, sys.stdin.readline().split())

    # Get the 0-indexed positions of the two persons from the dictionary
    # These lookups are efficient O(1) on average
    # Since A_i and B_i are guaranteed to be between 1 and N, and P_list contains
    # all numbers from 1 to N exactly once, person_A and person_B will always
    # be valid keys in the person_to_position dictionary.
    pos_A = person_to_position[person_A]
    pos_B = person_to_position[person_B]

    # Compare their 0-indexed positions to determine who is further to the front
    # A smaller index means the person is earlier in the P_list, thus further to the front
    if pos_A < pos_B:
        # person_A is at a smaller index (further front), print their number
        sys.stdout.write(str(person_A) + '
')
    else:
        # person_B is at a smaller index (further front)
        # Note: A and B are distinct person numbers (implied by A_i < B_i and unique P_i),
        # so their positions pos_A and pos_B will always be different.
        # Thus, if pos_A is not less than pos_B, it must be greater than pos_B.
        # print person_B's number
        sys.stdout.write(str(person_B) + '
')