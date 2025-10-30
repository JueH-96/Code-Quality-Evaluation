import sys

def count_full_moons(N, M, P):
    # Calculate the number of days between day 1 and day N, inclusive, on which he can see a full moon.
    count = (N - M) // P + 1
    return max(0, count)

# Read input from stdin
N, M, P = map(int, sys.stdin.readline().strip().split())

# Calculate the result
result = count_full_moons(N, M, P)

# Write the answer to stdout
sys.stdout.write(str(result) + '
')