import sys
import itertools

def solve(S, N):
    # Generate all possible binary strings by replacing '?' with '0' or '1'
    binary_strings = []
    for p in itertools.product('01', repeat=S.count('?')):
        binary_string = list(S)
        index = 0
        for i, c in enumerate(S):
            if c == '?':
                binary_string[i] = p[index]
                index += 1
        binary_strings.append(''.join(binary_string))

    # Convert binary strings to integers and filter out those greater than N
    integers = [int(s, 2) for s in binary_strings if int(s, 2) <= N]

    # Return the maximum integer if it exists, otherwise return -1
    return max(integers) if integers else -1

# Read input from stdin
S = input().strip()
N = int(input().strip())

# Solve the problem and write the answer to stdout
print(solve(S, N))