import sys

# Read input values
N, M, P = map(int, sys.stdin.readline().split())

# Calculate the number of full moon days
full_moon_days = (N - M) // P + 1 if M <= N else 0

# Print the result
print(full_moon_days)