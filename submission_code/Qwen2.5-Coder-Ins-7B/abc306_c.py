# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Find the middle index for each number
middle_indices = {}
for i in range(1, N + 1):
    indices = [j for j in range(3 * N) if A[j] == i]
    middle_indices[i] = indices[1]

# Sort numbers based on their middle indices
sorted_numbers = sorted(middle_indices, key=middle_indices.get)

# Print the sorted numbers
print(' '.join(map(str, sorted_numbers)))