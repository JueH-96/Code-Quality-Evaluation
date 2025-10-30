import sys

# Read input from stdin
input_values = [int(x) for x in input().split()]
A, B, C, D = input_values

# Count the frequency of each number
freq = {}
for num in [A, B, C, D]:
    freq[num] = freq.get(num, 0) + 1

# Check if a Full House can be formed
if len(freq) == 2 and (3 in freq.values() and 2 in freq.values()):
    print("Yes")
else:
    print("No")