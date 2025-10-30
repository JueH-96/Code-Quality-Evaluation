import sys

# Read N from the first line
N = int(sys.stdin.readline())

# Read the N integers from the second line
# sys.stdin.readline().split() reads the line and splits it into a list of strings
# map(int, ...) converts each string in the list to an integer
# list(...) converts the map object to a list
A = list(map(int, sys.stdin.readline().split()))

# Check if all elements in the list A are equal
# One way to do this is to convert the list to a set.
# If all elements are equal, the set will contain only one distinct element, so its size will be 1.
if len(set(A)) == 1:
    print("Yes")
else:
    print("No")