# YOUR CODE HERE
import sys

# Read the number of elements in the sequence.
N = int(sys.stdin.readline())

# Read the sequence of numbers A.
# Use sys.stdin.readline().split() to get the space-separated numbers from the input line.
# Use map(int, ...) to convert these string numbers into integers.
# Use list(...) to convert the map object into a list of integers.
A = list(map(int, sys.stdin.readline().split()))

# Initialize the result list B.
# B will have the same length as the input sequence A, which is N.
# We initialize all elements of B to -1, which is the required value if no previous occurrence is found.
B = [-1] * N

# Create a dictionary to keep track of the most recent 0-based index for each distinct number value encountered so far.
# This dictionary will help us efficiently find the largest index j < i such that A[j] == A[i].
# The keys of this dictionary will be the number values themselves (elements from sequence A).
# The values associated with these keys will be the 0-based index (i.e., 0, 1, ..., N-1) of the most recent position where that number value was seen.
last_seen = {}

# Iterate through the sequence A from the first element (at index 0) up to the last element (at index N-1).
# The variable 'i' represents the current 0-based index.
for i in range(N):
    current_value = A[i]

    # Check if the current_value has been seen before in the sequence at an index less than 'i'.
    # We can check this by seeing if current_value exists as a key in our 'last_seen' dictionary.
    if current_value in last_seen:
        # If current_value is found in 'last_seen', it means we have seen this value before.
        # The value stored in last_seen[current_value] is the 0-based index (let's call it j) of the *most recent* previous occurrence of current_value (where j < i).
        most_recent_0_based_index = last_seen[current_value]

        # The problem requires the output B_i to be the 1-based index of this most recent previous occurrence.
        # If the 0-based index is 'j', the corresponding 1-based index is 'j + 1'.
        # So, we set the i-th element of list B to this 1-based index.
        B[i] = most_recent_0_based_index + 1
    # If current_value is not in the 'last_seen' dictionary, it means this is the first time we are seeing this number value in the sequence up to the current index 'i'.
    # In this case, there is no index j < i such that A[j] == A[i].
    # According to the problem, B_i should be -1. Since we initialized the list B with -1s, B[i] already has the correct value, so we don't need an 'else' block here to set B[i] = -1.

    # After determining the value for B[i], we must update the 'last_seen' dictionary.
    # The current index 'i' is now the most recent 0-based index where 'current_value' was found.
    # Store this current index 'i' in the dictionary, associated with 'current_value'.
    # If 'current_value' was already in the dictionary, this operation updates its value to the new, more recent index 'i'.
    last_seen[current_value] = i

# After the loop has processed all elements from index 0 to N-1, the list B is populated with the desired results.
# The problem requires printing the elements of B separated by spaces.
# The '*' operator is used here to unpack the list B into individual arguments for the print function.
# print(*B) is equivalent to calling print(B[0], B[1], B[2], ..., B[N-1]).
# The default separator for print is a space.
print(*B)