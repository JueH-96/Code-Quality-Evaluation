import sys
from collections import defaultdict

# Read N
N = int(sys.stdin.readline())

# Read the sequence A
A = list(map(int, sys.stdin.readline().split()))

# Create a dictionary to store the indices of occurrences for each number.
# The key will be the number (1 to N), and the value will be a list of indices
# where that number appears in A.
# defaultdict(list) is used so that when we access a new key, an empty list is
# automatically created for it.
indices = defaultdict(list)

# Iterate through the sequence A with 0-based indexing
for j, value in enumerate(A):
    # Append the current index j to the list of indices for the number 'value'
    indices[value].append(j)

# Create a list of tuples, where each tuple is (middle_occurrence_index, number).
# We will sort this list based on the middle_occurrence_index.
middle_occurrences = []

# Iterate through numbers from 1 to N
for i in range(1, N + 1):
    # For each number i, indices[i] is a list of the three indices where it appeared in A.
    # Since we iterated through A from index 0 upwards, the list indices[i] is already sorted.
    # The middle occurrence's 0-based index is the second element (at index 1) in this list.
    middle_index = indices[i][1]
    
    # Append a tuple (middle_index, number) to our list.
    # We put the middle_index first because we want to sort by this value.
    middle_occurrences.append((middle_index, i))

# Sort the list of tuples. By default, sorted() sorts a list of tuples
# based on the first element of the tuples, then the second if the first are equal, and so on.
# This gives us the numbers sorted by their middle occurrence index.
sorted_by_middle_index = sorted(middle_occurrences)

# Extract the numbers (the second element of the tuples) from the sorted list.
# This gives us the final sequence of numbers sorted according to the problem criteria.
result_sequence = [item[1] for item in sorted_by_middle_index]

# Print the result sequence, with elements separated by spaces.
# The '*' operator unpacks the list elements as arguments to the print function.
print(*result_sequence)