# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Find the indices of the largest and second largest elements
first_max_index = A.index(max(A))
A[first_max_index] = float('-inf')  # Replace the largest element with negative infinity
second_max_index = A.index(max(A))

# The index of the second largest element is second_max_index + 1 (1-based index)
print(second_max_index + 1)