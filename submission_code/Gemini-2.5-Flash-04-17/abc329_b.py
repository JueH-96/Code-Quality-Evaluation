import sys

# Read N (not strictly needed for this approach, but read as per format)
# N = int(sys.stdin.readline())

# Read the list of integers from the second line
# Use map(int, ...) to convert each string to an integer
# Use split() to separate the space-separated numbers
# Convert the result to a list
A = list(map(int, sys.stdin.readline().split()))

# Get unique elements from the list
# Using set() automatically handles duplicates and gives unique values
unique_A = list(set(A))

# Sort the unique elements in descending order
# This places the largest unique element at index 0,
# the second largest unique element at index 1, and so on.
unique_A.sort(reverse=True)

# The problem asks for the largest among those integers that are not the largest.
# This is precisely the second largest unique element in the original list.
# Since the constraints guarantee that not all elements are equal (N >= 2 and
# not all A_i are equal), there are at least two unique elements.
# Therefore, unique_A[1] is guaranteed to exist and holds the correct answer.
print(unique_A[1])