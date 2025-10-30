# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
A = list(map(int, data[2:]))

# Sort the array to easily find the minimum and maximum values
A.sort()

# The minimum possible value of the difference between the maximum and minimum values of B
# is achieved by removing the K smallest and K largest elements
# and concatenating the remaining elements
result = A[N-K-1] - A[K]

print(result)