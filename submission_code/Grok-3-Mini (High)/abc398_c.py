import sys
from collections import Counter

# Read input from stdin
data = sys.stdin.read().split()
N = int(data[0])
A_values = list(map(int, data[1:]))

# Compute frequency of each value
freq = Counter(A_values)

# Initialize variables to keep track of the maximum unique value and its index
max_val = -1
max_index = -1

# Iterate through the list to find the maximum value with frequency 1
for i, val in enumerate(A_values):
    if freq[val] == 1:
        if val > max_val:
            max_val = val
            max_index = i

# Check if any unique value was found
if max_index == -1:
    print(-1)
else:
    # Print the 1-based index (label) of the person
    print(max_index + 1)