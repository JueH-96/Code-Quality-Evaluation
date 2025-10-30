import sys
import bisect

# Read all input and parse integers
data = list(map(int, sys.stdin.read().split()))

# Extract N and M
N = data[0]
M = data[1]

# Extract the list of fireworks days
fireworks_days = data[2:2 + M]

# For each day from 1 to N, compute the number of days until next fireworks
for day in range(1, N + 1):
    # Find the index of the smallest fireworks day >= day
    idx = bisect.bisect_left(fireworks_days, day)
    # The next fireworks day is at this index
    next_firework = fireworks_days[idx]
    # Compute the difference and print
    print(next_firework - day)