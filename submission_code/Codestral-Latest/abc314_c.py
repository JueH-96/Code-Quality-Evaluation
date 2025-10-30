# YOUR CODE HERE
import sys
from collections import defaultdict, deque

# Read input
N, M = map(int, sys.stdin.readline().split())
S = sys.stdin.readline().strip()
C = list(map(int, sys.stdin.readline().split()))

# Create a dictionary to store characters of each color
color_dict = defaultdict(deque)

# Populate the dictionary with characters
for i in range(N):
    color_dict[C[i]].append(S[i])

# Perform the right circular shift for each color
for i in range(1, M + 1):
    if color_dict[i]:
        last_char = color_dict[i].pop()
        color_dict[i].appendleft(last_char)

# Reconstruct the final string
final_S = []
for i in range(N):
    final_S.append(color_dict[C[i]].popleft())

# Print the final string
print(''.join(final_S))