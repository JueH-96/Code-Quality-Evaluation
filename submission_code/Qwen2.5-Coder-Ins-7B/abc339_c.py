# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Calculate the prefix sums
prefix_sums = [0] * (N + 1)
for i in range(N):
    prefix_sums[i + 1] = prefix_sums[i] + A[i]

# Find the minimum possible current number of passengers
min_passengers = min(prefix_sums)

# Print the result
print(min_passengers)