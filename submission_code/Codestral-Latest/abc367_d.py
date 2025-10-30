import sys
from itertools import accumulate

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:]))

# Calculate the cumulative sum of steps
cumulative_steps = list(accumulate(A))

# Dictionary to store the first occurrence of each remainder when divided by M
first_occurrence = {}
first_occurrence[0] = -1

count = 0

for i in range(N):
    remainder = cumulative_steps[i] % M
    if remainder in first_occurrence:
        count += i - first_occurrence[remainder]
    first_occurrence[remainder] = i

# For the case where we wrap around from the last rest area to the first
remainder = cumulative_steps[-1] % M
if remainder in first_occurrence:
    count += N - first_occurrence[remainder]

print(count)