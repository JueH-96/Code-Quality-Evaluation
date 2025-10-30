import sys
input = sys.stdin.read

# Read input
data = input().split()
A = list(map(int, data[:9]))
B = list(map(int, data[9:]))

# Calculate total scores
total_A = sum(A)
total_B = sum(B)

# Determine the minimum runs needed for Team Aoki to win
min_runs_needed = total_A - total_B + 1

# Print the result
print(min_runs_needed)