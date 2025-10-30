import sys

# Read input from stdin
data = sys.stdin.read().split()
N = int(data[0])
S = data[1]

# Initialize count of valid positions
count = 0

# Iterate through each possible index where i ranges from 0 to N-3 inclusive
for i in range(0, N - 2):
    # Check if seats i and i+2 are occupied and seat i+1 is unoccupied
    if S[i] == '#' and S[i + 1] == '.' and S[i + 2] == '#':
        count += 1

# Output the count
print(count)