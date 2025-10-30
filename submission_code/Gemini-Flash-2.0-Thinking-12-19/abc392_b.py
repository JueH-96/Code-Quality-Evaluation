# YOUR CODE HERE
import sys

# Read N and M from the first line of input
line1 = sys.stdin.readline().split()
n = int(line1[0])
m = int(line1[1])

# Read the M elements of A from the second line
# Convert them to integers and store them in a set for efficient membership testing
# Using a set provides O(1) average time complexity for checking if a number is present.
# Given 1 <= M <= N, the second line will always have M elements to read.
line2 = sys.stdin.readline().split()
a_elements = set(map(int, line2))

# Find the numbers between 1 and N (inclusive) that are not in A
missing_numbers = []
# Iterate through all integers from 1 up to N
for i in range(1, n + 1):
    # Check if the current integer i is NOT present in the set of A elements
    if i not in a_elements:
        # If it's not in A, add it to our list of missing numbers
        # Since we iterate from 1 to N, the numbers are added in ascending order.
        missing_numbers.append(i)

# Output the count of missing numbers
print(len(missing_numbers))

# Output the missing numbers space-separated
# The '*' operator unpacks the list 'missing_numbers' into arguments for the print function.
# By default, print separates arguments with a space.
# If missing_numbers is empty (e.g., for N=6, M=6 in Sample 2), print(*[]) effectively calls print(),
# which outputs just a newline, leaving the second line empty as required.
print(*missing_numbers)