# YOUR CODE HERE
import sys

input = sys.stdin.read().split()
N = int(input[0])
A = list(map(int, input[1:]))

# Find the largest and second largest elements
largest = second_largest = float('-inf')
for i in range(N):
    if A[i] > largest:
        second_largest = largest
        largest = A[i]
    elif A[i] > second_largest:
        second_largest = A[i]

# Find the index of the second largest element
second_largest_index = A.index(second_largest) + 1

print(second_largest_index)