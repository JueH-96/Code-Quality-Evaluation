import sys

# Read input
strings = [sys.stdin.readline().strip() for _ in range(12)]

# Count the number of strings where the length of S_i is i
count = sum(1 for i, s in enumerate(strings, start=1) if len(s) == i)

# Print the result
print(count)