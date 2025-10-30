# YOUR CODE HERE

# Read the inputs
N, M, P = map(int, input().split())

# Calculate the number of days between day 1 and day N, inclusive, on which he can see a full moon
num_days = (N - M) // P + 1

# Print the answer
print(num_days)