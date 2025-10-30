# YOUR CODE HERE
# Read the three integer inputs: first term A, last term B, and common difference D.
# input().split() reads a line and splits it into a list of strings by whitespace.
# map(int, ...) applies the int() function to each item in the list, converting them to integers.
A, B, D = map(int, input().split())

# The range(start, stop, step) function generates an arithmetic progression.
# We set start=A, stop=B+1 (as range is exclusive), and step=D.
# The problem guarantees that a valid sequence exists, so B is reachable from A with step D.
sequence_generator = range(A, B + 1, D)

# The * operator unpacks the items from the sequence_generator.
# For example, if the sequence is (3, 5, 7, 9), print(*sequence) is equivalent to print(3, 5, 7, 9).
# By default, print() separates arguments with a space, which is the desired output format.
print(*sequence_generator)