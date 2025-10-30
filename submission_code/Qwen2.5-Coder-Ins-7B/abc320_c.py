import sys

# Read input from stdin
input = sys.stdin.read
data = input().split()

# Parse input
M = int(data[0])
S1 = data[1]
S2 = data[2]
S3 = data[3]

# Initialize variables
min_time = float('inf')

# Iterate over all possible characters
for char in '0123456789':
    # Check if the current character is present on all three reels
    if char in S1 and char in S2 and char in S3:
        # Calculate the time required to stop all reels at the current character
        time1 = (S1.index(char) + 1) % M
        time2 = (S2.index(char) + 1) % M
        time3 = (S3.index(char) + 1) % M
        max_time = max(time1, time2, time3)
        # Update the minimum time if the current time is smaller
        if max_time < min_time:
            min_time = max_time

# If it is impossible to stop all reels at the same character, print -1
if min_time == float('inf'):
    print(-1)
else:
    print(min_time)