import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
S = list(data[2])
C = list(map(int, data[3:]))

# Create a dictionary to store the indices of each color
color_indices = {}
for i in range(1, M + 1):
    color_indices[i] = []

for i in range(N):
    color_indices[C[i]].append(i)

# Perform the right circular shift for each color
for i in range(1, M + 1):
    indices = color_indices[i]
    last_char = S[indices[-1]]
    for j in range(len(indices) - 1, 0, -1):
        S[indices[j]] = S[indices[j - 1]]
    S[indices[0]] = last_char

# Print the final string
print(''.join(S))