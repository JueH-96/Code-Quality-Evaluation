import sys

# Read inputs
N, T, P = map(int, sys.stdin.readline().split())
hair_lengths = list(map(int, sys.stdin.readline().split()))

# Initialize days and count
days = 0
count = 0

# Check each day
while count < P:
    days += 1
    count = sum(1 for length in hair_lengths if length + days >= T)

# Print the result
print(days)