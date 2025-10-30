import sys

# Read all input and split into a list
data = sys.stdin.read().split()

# Parse N, T, P from the first three elements
N = int(data[0])
T = int(data[1])
P = int(data[2])

# Parse the hair lengths from the next N elements
hair_lengths = [int(x) for x in data[3:3+N]]

# Calculate the number of days for each person to reach at least T
reach_days = [max(0, T - length) for length in hair_lengths]

# Sort the reach days
sorted_reach_days = sorted(reach_days)

# The P-th smallest reach day is the answer
days = sorted_reach_days[P-1]

# Print the result
print(days)