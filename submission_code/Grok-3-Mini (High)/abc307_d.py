import sys

# Read input
N = int(sys.stdin.readline().strip())
S = sys.stdin.readline().strip()

# Find indices where there are parentheses
paren_idx = [i for i in range(N) if S[i] in '()']

# Use stack to find matched pairs
stack = []
matched_pairs = []
for idx in paren_idx:
    if S[idx] == '(':
        stack.append(idx)
    else:  # ')'
        if stack:
            left = stack.pop()
            matched_pairs.append((left, idx))

# Sort the matched pairs by start index
intervals = matched_pairs
intervals.sort()

# Merge overlapping or adjacent intervals
if intervals:
    merged = [(intervals[0][0], intervals[0][1])]
    for L, R in intervals[1:]:
        last_L, last_R = merged[-1]
        if last_R + 1 >= L:  # Adjacent or overlapping
            merged[-1] = (last_L, max(last_R, R))
        else:
            merged.append((L, R))
else:
    merged = []

# Create a keep list, initially all True
keep = [True] * N

# Set keep to False for positions in the merged intervals
for L, R in merged:
    for k in range(L, R + 1):
        keep[k] = False

# Build the result string with kept characters
result = ''.join(S[k] for k in range(N) if keep[k])

# Print the result
print(result)