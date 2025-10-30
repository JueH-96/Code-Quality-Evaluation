# YOUR CODE HERE
import sys
from collections import Counter

# Read N, the number of people
N = int(sys.stdin.readline())

# Read the list of integers A_1, A_2, ..., A_N
# Store them in a list A, where A[0] is A_1, A[1] is A_2, ..., A[i-1] is A_i.
# Person i is associated with the value A_i, which is stored at index i-1 in list A.
A = list(map(int, sys.stdin.readline().split()))

# Count occurrences of each number in the list A
# The counts dictionary will store {value: number_of_occurrences}
counts = Counter(A)

# Find the maximum value among the numbers that appear exactly once
# Initialize max_unique_value. According to constraints, A_i >= 1, so -1
# is a safe initial value to indicate that no unique value has been found yet.
max_unique_value = -1

# Iterate through the items (value, count pairs) in the counts dictionary
for value, count in counts.items():
    # Check if the current value appeared exactly once
    if count == 1:
        # If it's a unique value, check if it's greater than the maximum unique
        # value found so far.
        if value > max_unique_value:
            # Update max_unique_value if the current unique value is larger
            max_unique_value = value

# After iterating through all counted values, max_unique_value holds the largest
# value that is unique in the input list, or remains -1 if no such value exists.

# Check if any unique value was found
if max_unique_value == -1:
    # If max_unique_value is still -1, it means no number in the input list
    # appeared exactly once. In this case, no person satisfies the condition.
    # Print -1 as required.
    print(-1)
else:
    # If a maximum unique value was found, we need to find the label (1-based index)
    # of the person who has this specific value.
    # Iterate through the original list A with index i from 0 to N-1.
    # A[i] holds the value for the person labeled i + 1.
    # We are looking for the index i such that A[i] is equal to max_unique_value.
    # The corresponding person's label is i + 1.
    found_label = -1 # Initialize label variable

    # Since max_unique_value is unique across the entire list A, there is guaranteed
    # to be exactly one index i where A[i] == max_unique_value.
    for i in range(N):
        # Check if the value at the current index is the max unique value
        if A[i] == max_unique_value:
            # If it is, the label of this person is i + 1
            found_label = i + 1
            # We found the person, so we can stop searching
            break

    # Print the label of the person who has the greatest unique integer
    print(found_label)