import sys
from itertools import accumulate

input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
medicines = [(int(data[2*i+2]), int(data[2*i+3])) for i in range(N)]

# Sort medicines by the number of days they need to be taken
medicines.sort()

# Calculate the cumulative sum of pills for each day
cumulative_pills = [0] * (medicines[-1][0] + 1)
for a, b in medicines:
    cumulative_pills[0] += b
    if a < len(cumulative_pills):
        cumulative_pills[a] -= b

# Calculate the prefix sum of cumulative_pills
prefix_sum = list(accumulate(cumulative_pills))

# Find the first day where the number of pills is less than or equal to K
for day, pills in enumerate(prefix_sum, start=1):
    if pills <= K:
        print(day)
        break