import sys

# Read N and L from the first line
n, l = map(int, sys.stdin.readline().split())

# Read the scores from the second line
scores = list(map(int, sys.stdin.readline().split()))

# Initialize a counter for passed people
passed_count = 0

# Iterate through the scores and count how many are at least L
for score in scores:
    if score >= l:
        passed_count += 1

# Print the result
print(passed_count)