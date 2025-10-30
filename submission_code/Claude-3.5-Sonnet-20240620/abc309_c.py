# YOUR CODE HERE
import sys
from collections import defaultdict

# Read input
N, K = map(int, input().split())
medicines = [list(map(int, input().split())) for _ in range(N)]

# Create a dictionary to store the number of pills for each day
pills_per_day = defaultdict(int)

# Calculate the number of pills for each day
for a, b in medicines:
    pills_per_day[1] += b
    pills_per_day[a + 1] -= b

# Find the first day with K pills or less
total_pills = 0
for day in range(1, 10**9 + 1):
    total_pills += pills_per_day[day]
    if total_pills <= K:
        print(day)
        break