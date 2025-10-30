# YOUR CODE HERE
import sys

# Read N
N = int(sys.stdin.readline())

# Read array A
# Using list comprehension might be slightly faster than map+list, but map+list is common.
# A = [int(x) for x in sys.stdin.readline().split()]
A = list(map(int, sys.stdin.readline().split()))

# Dictionary to store the last seen index for each number
# key: number, value: last seen index
last_seen = {}

# Initialize minimum length to infinity
# We are looking for the shortest subarray, so a very large initial value is needed
# float('inf') is a standard way to represent mathematical infinity in Python
min_length = float('inf')

# Iterate through the array with index i
for i in range(N):
    current_value = A[i]

    # Check if the current value has been seen before
    if current_value in last_seen:
        # Get the index of the previous occurrence of this value
        prev_index = last_seen[current_value]

        # Calculate the length of the contiguous subarray ending at index i
        # that contains the value A[i] and its previous occurrence at prev_index.
        # This subarray is A[prev_index ... i]. Its length is i - prev_index + 1.
        # This subarray is guaranteed to contain a repeated value (A[prev_index] == A[i]).
        current_subarray_length = i - prev_index + 1

        # Update the minimum length found so far with this new length.
        # We want the shortest such subarray among all possibilities.
        min_length = min(min_length, current_subarray_length)

        # Update the last seen index for the current value to the current index i.
        # When we encounter this value again later (at index j > i),
        # we want to calculate the distance between the occurrence at j
        # and the most recent occurrence before j, which is at index i.
        # This helps us find the minimum distance between *consecutive* occurrences
        # of any value in the array, which directly corresponds to the length of
        # the shortest subarray containing just two occurrences of that value.
        last_seen[current_value] = i

    else:
        # If the current value has not been seen before, record its index.
        # This is the first occurrence we've found so far.
        last_seen[current_value] = i

# After iterating through the entire array, check if min_length was updated.
# If min_length is still float('inf'), it means no repeated value was found
# in the array, and thus no subarray with a repeated value exists.
if min_length == float('inf'):
    # As per the problem statement, print -1 if no such subarray exists.
    print(-1)
else:
    # Otherwise, min_length holds the length of the shortest subarray found
    # that contains a repeated value.
    print(min_length)