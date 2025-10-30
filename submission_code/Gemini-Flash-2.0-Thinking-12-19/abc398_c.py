# YOUR CODE HERE
import sys
from collections import Counter

# Read input
N = int(sys.stdin.readline())
A = list(map(int, sys.stdin.readline().split()))

# Calculate frequencies of each value
counts = Counter(A)

# Find values that appear exactly once
# These are the values held by people who satisfy the uniqueness condition
unique_values = [value for value, count in counts.items() if count == 1]

# If no value appears exactly once, it means no person satisfies the condition
if not unique_values:
    print(-1)
else:
    # Among the unique values, find the maximum one
    max_unique_value = max(unique_values)

    # Find the index (0-based) in the original list A where this maximum unique value occurs.
    # Since the value is unique across the list, it will appear exactly once.
    # A.index(value) finds the index of the first occurrence, which is the only occurrence here.
    person_index = A.index(max_unique_value)

    # The person's label is the 1-based index
    print(person_index + 1)