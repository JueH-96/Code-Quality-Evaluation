# Read the input
N = int(input())
A = list(map(int, input().split()))

# Initialize a dictionary to store the indices of each element
indices = {}

# Initialize the minimum length of the subarray with a repeated value
min_length = float('inf')

# Iterate through the sequence
for i in range(N):
    # If the current element has been seen before
    if A[i] in indices:
        # Calculate the length of the subarray
        length = i - indices[A[i]] + 1
        # Update the minimum length if necessary
        min_length = min(min_length, length)
    # Store the index of the current element
    indices[A[i]] = i

# If no subarray with a repeated value was found, print -1
if min_length == float('inf'):
    print(-1)
# Otherwise, print the minimum length
else:
    print(min_length)