# YOUR CODE HERE
import sys

# Read N and M from the first line of input
line1 = sys.stdin.readline().split()
N = int(line1[0])
M = int(line1[1])

# Read the sequence A from the second line of input
# Constraint 1 <= M <= N, so A will always have M elements.
line2 = sys.stdin.readline().split()
A = list(map(int, line2))

# Store elements of A in a set for efficient lookup.
# Using a set allows checking for membership in O(1) on average.
set_A = set(A)

# Find integers between 1 and N (inclusive) that are not in A.
missing_integers = []
# Iterate through all integers from 1 up to N.
for i in range(1, N + 1):
    # Check if the current integer 'i' is not present in the set A.
    if i not in set_A:
        # If 'i' is not present in A, add it to our list of missing integers.
        # Since we are iterating in ascending order (1, 2, ..., N),
        # the missing_integers list will automatically be in ascending order.
        missing_integers.append(i)

# Print the count of missing integers.
print(len(missing_integers))

# Print the missing integers if the list is not empty.
# The list 'missing_integers' is already sorted.
# The '*' operator unpacks the list elements as arguments to print(),
# which prints them separated by spaces by default.
if missing_integers:
    print(*missing_integers)
# If 'missing_integers' is an empty list (i.e., len is 0), the 'if' condition
# is false, and this print statement is skipped. This results in an empty
# second line of output, which is the correct behavior when C=0.