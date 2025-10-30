# YOUR CODE HERE
import sys

# Read the input values A, B, D from standard input
# input().split() reads the line and splits it into strings based on whitespace
# map(int, ...) converts each string part into an integer
a, b, d = map(int, sys.stdin.readline().split())

# Generate the arithmetic sequence using the range function.
# range(start, stop, step) generates numbers starting from 'start',
# incrementing by 'step', and stopping *before* 'stop'.
# Since we want to include the last term B, we need the range to go up to B + 1.
# The problem guarantees that B is reachable from A with difference D,
# meaning B = A + k*D for some non-negative integer k.
# Therefore, the range function will include B as the last element.
sequence_generator = range(a, b + 1, d)

# Convert each number in the generated sequence to its string representation.
# map(str, sequence_generator) applies the str() function to each element lazily.
sequence_str = map(str, sequence_generator)

# Join the string representations of the numbers with a space in between.
# " ".join(...) consumes the iterator from map and concatenates the strings.
output_string = " ".join(sequence_str)

# Print the resulting string to standard output.
print(output_string)