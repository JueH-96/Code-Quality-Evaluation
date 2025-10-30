# YOUR CODE HERE
import sys

input = sys.stdin.read().split()
N = int(input[0])
S = input[1]
Q = int(input[2])
operations = input[3:]

# Create a mapping for the final replacements
mapping = {}
for i in range(Q):
    c, d = operations[i * 2], operations[i * 2 + 1]
    # If c is already mapped, we need to map it to the final destination
    while c in mapping:
        c = mapping[c]
    mapping[c] = d

# Apply the final mappings to the string
result = []
for char in S:
    final_char = char
    while final_char in mapping:
        final_char = mapping[final_char]
    result.append(final_char)

print(''.join(result))