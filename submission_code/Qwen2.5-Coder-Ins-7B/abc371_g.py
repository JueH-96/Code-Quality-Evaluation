# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
P = list(map(int, data[1:N+1]))
A = list(map(int, data[N+1:2*N+1]))

# Create a dictionary to store the index of each element in A
index_map = {A[i]: i for i in range(N)}

# Create a list to store the result
result = [0] * N

# Iterate over each element in P
for i in range(N):
    # Get the index of the current element in A
    current_index = index_map[P[i]]
    # Set the result at the current index to the current element
    result[current_index] = A[i]
    # Update the index map to reflect the new position of the current element
    index_map[P[i]] = i

# Print the result
print(' '.join(map(str, result)))