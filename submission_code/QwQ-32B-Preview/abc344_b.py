import sys

# Read all lines from standard input
lines = sys.stdin.read().splitlines()

# Convert each line to integer
nums = [int(line) for line in lines]

# Reverse the list
reversed_nums = nums[::-1]

# Print each element on a new line
for num in reversed_nums:
    print(num)