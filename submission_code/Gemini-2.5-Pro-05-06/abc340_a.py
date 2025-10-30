# Read the input values for A, B, and D
A, B, D = map(int, input().split())

# An arithmetic sequence starts with A. Each subsequent term is obtained by adding D to the previous term.
# The sequence stops when the term exceeds B. Since it's guaranteed that a sequence exists
# with B as the last term, we know that B will be exactly A + k*D for some non-negative integer k.

# Python's range(start, stop, step) function is suitable here.
# It generates numbers from 'start' up to (but not including) 'stop', with an increment of 'step'.
# To include B as the last term, the 'stop' value for range should be B + 1 (or anything up to B + D).
# For example, if A=3, B=9, D=2, then range(3, 9 + 1, 2) which is range(3, 10, 2)
# will generate 3, 5, 7, 9. The next term would be 11, which is not less than 10, so it stops.
# If A=10, B=10, D=1, then range(10, 10 + 1, 1) which is range(10, 11, 1)
# will generate 10. The next term would be 11, which is not less than 11, so it stops.

# Generate the sequence of numbers
sequence_numbers = range(A, B + 1, D)

# Convert each number in the sequence to a string
# map(str, sequence_numbers) creates an iterator that applies str() to each item from sequence_numbers
sequence_strings = map(str, sequence_numbers)

# Join the string representations of the numbers with a space in between
output_string = " ".join(sequence_strings)

# Print the resulting string
print(output_string)