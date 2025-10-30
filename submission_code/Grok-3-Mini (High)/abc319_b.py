import sys

# Read input from stdin
data = sys.stdin.read().strip()
N = int(data)

# Find all divisors j of N between 1 and 9
divs = [j for j in range(1, 10) if N % j == 0]

# Build the result string
result = []
for i in range(N + 1):
    try:
        # Find the minimum j such that i is a multiple of N//j
        min_j = min(j for j in divs if i % (N // j) == 0)
        result.append(str(min_j))
    except ValueError:
        # No such j exists, append '-'
        result.append('-')

# Output the result string
print(''.join(result))