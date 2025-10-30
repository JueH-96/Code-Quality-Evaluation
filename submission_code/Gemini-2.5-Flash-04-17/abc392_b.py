# YOUR CODE HERE
import sys

# Read N and M
# N: the upper bound of the range of integers [1, N]
# M: the number of elements in the sequence A
n, m = map(int, sys.stdin.readline().split())

# Read the M integers of sequence A from the second line.
# Convert these integers into a set for efficient membership testing (checking if a number is present in A).
# Using a set allows for O(1) average time complexity lookup.
# Since the constraint is 1 <= M, there will always be M integers to read.
a_present = set(map(int, sys.stdin.readline().split()))

# Initialize an empty list to store the integers between 1 and N that are not found in A.
missing_numbers = []

# Iterate through each integer from 1 up to N (inclusive).
for i in range(1, n + 1):
    # Check if the current integer 'i' is NOT present in the set 'a_present'.
    if i not in a_present:
        # If 'i' is not found in A, add it to our list of missing numbers.
        # Since we iterate from 1 upwards, the numbers are added to the list in ascending order.
        missing_numbers.append(i)

# Print the count of missing integers. This is the length of the 'missing_numbers' list.
print(len(missing_numbers))

# Print the list of missing integers.
# If the 'missing_numbers' list is not empty, unpack its elements using the * operator
# and print them separated by spaces (the default print separator).
# If the list is empty (meaning all numbers from 1 to N were in A), the 'if' condition is false,
# and nothing is printed on the second line, fulfilling the requirement for C=0.
if missing_numbers:
    print(*missing_numbers)