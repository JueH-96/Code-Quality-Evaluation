# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Sort the array to make it easier to calculate the maximum difference
A.sort()

# Initialize the result
result = 0

# Iterate over each element in the array
for i in range(N):
    # For each element, calculate the number of elements greater than it
    count = N - i - 1
    # Multiply the count by the element to get the contribution to the result
    result += A[i] * count

# Print the result
print(result)