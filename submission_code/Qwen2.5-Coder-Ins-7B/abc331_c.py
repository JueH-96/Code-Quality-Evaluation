# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Initialize the result list
B = [0] * N

# Calculate the sum of all elements in A
total_sum = sum(A)

# Iterate through each element in A
for i in range(N):
    # Subtract the current element from the total sum
    total_sum -= A[i]
    # The remaining sum is the sum of elements greater than A[i]
    B[i] = total_sum

# Print the result
print(' '.join(map(str, B)))